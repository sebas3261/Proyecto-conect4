import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy
from typing import override


class UCB1Policy(Policy):

    def __init__(self):
        self.Q = {}          # Q(s,a)
        self.N_s = {}        # visitas del estado
        self.N_sa = {}       # visitas del par (s,a)

        self.memory = []     # historial en la partida
        self.alpha = 0.2
        self.c = 2.0

        self.rng = np.random.default_rng()

        self._load_qvalues()

    # ---------------------------------------------------------
    @override
    def mount(self, time_out=None):
        self.memory.clear()

    # ---------------------------------------------------------
    @override
    def act(self, s: np.ndarray) -> int:
        self._normalize(s)
        state_key = self._state_key(s)

        # columnas válidas
        available = np.flatnonzero(s[0] == 0)
        if available.size == 0:
            return 0

        # visitas del estado
        n_s = self.N_s.get(state_key, 0) + 1
        self.N_s[state_key] = n_s
        log_ns = np.log(n_s + 1)

        best_value = -1e18
        best_a = int(available[0])

        for a in available:
            key_sa = f"{state_key}|{int(a)}"
            q = self.Q.get(key_sa, 0.0)
            n_sa = self.N_sa.get(key_sa, 0)

            # UCB1
            bonus = self.c * np.sqrt(log_ns / (n_sa + 1))
            value = q + bonus

            if value > best_value:
                best_value = value
                best_a = int(a)

        # registrar visita a (s,a)
        key_best = f"{state_key}|{best_a}"
        self.N_sa[key_best] = self.N_sa.get(key_best, 0) + 1

        # guardar memoria para actualizar al final
        self.memory.append((state_key, best_a))

        return best_a

    # ---------------------------------------------------------
    @override
    def final(self, reward):
        # actualizar Q-values (Q-learning)
        for state_key, action in self.memory:
            key_sa = f"{state_key}|{action}"
            old_q = self.Q.get(key_sa, 0.0)
            self.Q[key_sa] = old_q + self.alpha * (reward - old_q)

        self.memory.clear()
        # ❗ NO guardar aquí (solo train_mp guarda para evitar lag)

    # =========================================================
    # Utils
    # =========================================================

    def _normalize(self, s):
        flat = s.reshape(-1)
        if np.count_nonzero(flat == -1) > np.count_nonzero(flat == 1):
            s *= -1

    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    # ruta base del qvalues
    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues_ucb.json")

    # cargar Q
    def _load_qvalues(self):
        path = self._json_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                text = f.read().strip()
                self.Q = json.loads(text) if text else {}
        except:
            self.Q = {}

    # ---------------------------------------------------------
    # SAVE compatible con MULTIPROCESO
    # ---------------------------------------------------------
    def _save_qvalues(self, path_override=None):
        """
        Soporta path_override para que cada proceso guarde:
          qvalues_ucb_worker0.json
          qvalues_ucb_worker1.json
          ...
        """
        path = path_override or self._json_path()
        tmp = path + ".tmp"

        # escribir JSON temporal
        with open(tmp, "w") as f:
            json.dump(self.Q, f)

        # reemplazo seguro
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Windows fix
            import time
            time.sleep(0.05)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
