import numpy as np
import json
import os
from connect4.policy import Policy
from typing import override
from math import log, sqrt


class UCB1Normalized(Policy):

    def __init__(self):
        # Diccionario Q(s,a)
        # formato: Q["state_hex|col"] = valor
        self.Q = {}

        # Conteos para UCB1
        # formato: N[(state_hex, col)] = veces elegida la acción
        self.N = {}
        # formato: N_state[state_hex] = veces visto el estado
        self.N_state = {}

        # Parámetros
        self.alpha = 0.20
        self.c = 2.0            # constante UCB1
        self.rng = np.random.default_rng()

        # Memoria para MC update
        self.memory = []  # (state_key, action)

        # Cargar Q-values
        self._load_qvalues()

    # ============================================================
    # MOUNT — reset temporal
    # ============================================================
    @override
    def mount(self, time_out=None):
        self.memory.clear()

    # ============================================================
    # NORMALIZAR (no muta el original)
    # ============================================================
    def _normalize(self, board: np.ndarray) -> np.ndarray:
        b = board.copy()
        ones = np.count_nonzero(b == 1)
        negs = np.count_nonzero(b == -1)
        if negs > ones:
            b = -b
        return b

    # ============================================================
    # ACT — selección UCB1
    # ============================================================
    @override
    def act(self, board: np.ndarray) -> int:

        # Copiar tabla real del entorno
        b_real = board.copy()
        b = self._normalize(b_real)

        # Acciones posibles (en el tablero real)
        available = np.flatnonzero(b_real[0] == 0)
        if available.size == 0:
            return 0

        # Clave del estado (normalizado)
        state_key = b.tobytes().hex()

        # Inicializar conteos
        if state_key not in self.N_state:
            self.N_state[state_key] = 0

        # Inicializar Q y N para cada acción
        for c in available:
            key = (state_key, int(c))
            if key not in self.N:
                self.N[key] = 0
            if f"{state_key}|{c}" not in self.Q:
                self.Q[f"{state_key}|{c}"] = 0.0

        # Aumentar cuenta de estado
        self.N_state[state_key] += 1
        total_visits = self.N_state[state_key]

        # UCB1
        def ucb1_value(c):
            key = (state_key, int(c))
            q = self.Q[f"{state_key}|{c}"]
            n = self.N[key]

            if n == 0:
                return float("inf")  # asegurar explorar todas

            return q + self.c * sqrt(log(total_visits) / n)

        # Seleccionar mejor acción
        best_action = max(available, key=ucb1_value)
        best_action = int(best_action)

        # Registrar para MC update
        self.memory.append((state_key, best_action))

        # Incrementar conteo para UCB1
        self.N[(state_key, best_action)] += 1

        return best_action

    # ============================================================
    # FINAL — MC update
    # ============================================================
    @override
    def final(self, reward: int):

        for state_key, action in self.memory:
            qkey = f"{state_key}|{action}"
            old = self.Q.get(qkey, 0.0)
            self.Q[qkey] = old + self.alpha * (reward - old)

        self.memory.clear()
        self._save_qvalues()

    # ============================================================
    # Q-values file utilities
    # ============================================================
    def _json_path(self):
        """
        Ruta ABSOLUTA al archivo qvalues.json
        en el MISMO folder que policy.py.
        """
        base = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(base, "qvalues_ucb1.json")

    def _load_qvalues(self):
        path = self._json_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.Q = data.get("Q", {})
            self.N = {tuple(k.split("|")): v for k, v in data.get("N", {}).items()}
            self.N_state = data.get("N_state", {})

        except Exception:
            self.Q = {}
            self.N = {}
            self.N_state = {}

    def _save_qvalues(self):
        path = self._json_path()
        tmp = path + ".tmp"

        data = {
            "Q": self.Q,
            "N": {f"{k[0]}|{k[1]}": v for k, v in self.N.items()},
            "N_state": self.N_state
        }

        with open(tmp, "w") as f:
            json.dump(data, f)

        try:
            os.replace(tmp, path)
        except:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
