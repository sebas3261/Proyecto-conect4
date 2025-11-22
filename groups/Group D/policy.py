import numpy as np
import json
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override


class TrialBasedOPIPolicy(Policy):

    def __init__(self):
        self.Q = {}                  # Q(s,a)
        self.alpha = 0.15            # factor de aprendizaje
        self.rollout_depth = 3       # qué tan profundo simula
        self.memory = []             # historial de la partida

        self.rng = np.random.default_rng()

        self._load_qvalues()

    # ---------------------------------------------------------
    @override
    def mount(self, time_out=None):
        self.memory.clear()

    # ---------------------------------------------------------
    @override
    def act(self, s: np.ndarray) -> int:
        # Normalizar (si somos -1 convertimos a 1)
        self._normalize(s)
        state_key = self._state_key(s)

        # columnas válidas
        available = np.flatnonzero(s[0] == 0)
        if available.size == 0:
            return 0

        best_val = -1e18
        best_act = int(available[0])

        # evaluar cada acción con rollout
        for a in available:
            val = self._estimate_action_value(s, int(a))
            if val > best_val:
                best_val = val
                best_act = int(a)

        # update on-policy
        key = f"{state_key}|{best_act}"
        old_q = self.Q.get(key, 0.0)
        self.Q[key] = old_q + self.alpha * (best_val - old_q)

        self.memory.append((state_key, best_act))

        return best_act

    # ---------------------------------------------------------
    # Estimar valor de acción usando rollout MC
    # ---------------------------------------------------------
    def _estimate_action_value(self, s, a):
        state = ConnectState(s)
        nxt = state.transition_fast(a)

        if nxt.is_final():
            return nxt.get_winner()

        return self._rollout(nxt, self.rollout_depth)

    def _rollout(self, state, depth):
        # límite de profundidad o final
        if depth == 0 or state.is_final():
            return state.get_winner()

        available = np.flatnonzero(state.board[0] == 0)
        if available.size == 0:
            return 0

        a = int(self.rng.choice(available))
        nxt = state.transition_fast(a)

        return self._rollout(nxt, depth - 1)

    # ---------------------------------------------------------
    @override
    def final(self, reward):
        # Esta policy NO usa reward final directamente
        self.memory.clear()
        # ❗ NO guardar aquí → train_mp controla persistencia

    # ============================================================
    # Utils
    # ============================================================

    def _normalize(self, s):
        flat = s.reshape(-1)
        if np.count_nonzero(flat == -1) > np.count_nonzero(flat == 1):
            s *= -1

    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    # ruta base
    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues_opi.json")

    # ---------------------------------------------------------
    # Cargar Q
    # ---------------------------------------------------------
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
    # Guardado compatible MULTICORE
    # ---------------------------------------------------------
    def _save_qvalues(self, path_override=None):
        """
        Guardado seguro y compatible con entrenamiento en paralelo.
        Permite usar archivos:
            qvalues_opi_worker0.json
            qvalues_opi_worker1.json
            ...
        """
        path = path_override or self._json_path()
        tmp = path + ".tmp"

        # escribir archivo temporal
        with open(tmp, "w") as f:
            json.dump(self.Q, f)

        # reemplazo atómico seguro
        try:
            os.replace(tmp, path)
        except PermissionError:
            # fix para Windows
            import time
            time.sleep(0.05)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
