import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy


class UncertaintyWithEGreedy(Policy):
    """
    Policy B: e-greedy con Q-learning.
    Versión arreglada: bootstrapping completo + guardado en cada paso.
    """

    def __init__(self):
        self.Q: dict[str, float] = {}
        self.memory: list[tuple[str, int]] = []

        # Hiperparámetros
        self.epsilon: float = 0.01   # exploración mínima
        self.alpha: float = 0.2
        self.time_out = None

        self._load_qvalues()

    def mount(self, time_out=None):
        self.time_out = max(1, int(time_out)) if time_out is not None else None

    # =========================================================
    # ACT — e-greedy + bootstrapping completo + auto-save
    # =========================================================
    def act(self, s: np.ndarray) -> int:
        s = s.copy()
        self._normalize(s)

        state_key = self._state_key(s)
        available_cols = [c for c in range(7) if s[0, c] == 0]

        if not available_cols:
            return 0  # seguridad

        rng = np.random.default_rng()

        # 🔥 BOOTSTRAP Q(s,a) PARA TODAS LAS ACCIONES POSIBLES
        for a in available_cols:
            key = f"{state_key}|{a}"
            if key not in self.Q:
                self.Q[key] = 0.0

        # --- Epsilon-greedy ---
        if rng.random() < self.epsilon:
            action = int(rng.choice(available_cols))
        else:
            action = self._exploit(state_key, available_cols)

        # Registrar transición
        self.memory.append((state_key, action))

        # 🔥 GUARDAR INMEDIATAMENTE (como UCB1)
        self._save_qvalues()

        return action

    # =========================================================
    # EXPLOIT — elegir mejor acción según Q
    # =========================================================
    def _exploit(self, state_key: str, available_cols: list[int]) -> int:
        best_q = -1e9
        best_action = available_cols[0]

        for a in available_cols:
            q = self.Q.get(f"{state_key}|{a}", 0.0)
            if q > best_q:
                best_q = q
                best_action = a

        return best_action

    # =========================================================
    # FINAL — Q-learning + auto-save
    # =========================================================
    def final(self, reward: float):
        if not self.memory:
            return

        for state_key, action in self.memory:
            key = f"{state_key}|{action}"
            old_q = self.Q.get(key, 0.0)
            new_q = old_q + self.alpha * (reward - old_q)
            self.Q[key] = new_q

        self.memory = []

        # 🔥 GUARDAR SIEMPRE
        self._save_qvalues()

    # =========================================================
    # NORMALIZACIÓN
    # =========================================================
    def _normalize(self, s):
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)

        if ones == negs:
            return
        if negs == ones + 1:
            s[:] = -s
            return
        if negs > ones:
            s[:] = -s

    # =========================================================
    # KEYS
    # =========================================================
    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues.json")

    # =========================================================
    # LOAD robusto
    # =========================================================
    def _load_qvalues(self):
        path = self._json_path()
        if not os.path.exists(path):
            self.Q = {}
            return

        try:
            with open(path, "r") as f:
                text = f.read().strip()

            if not text:
                self.Q = {}
                return

            self.Q = json.loads(text)
        except Exception:
            self.Q = {}

    # =========================================================
    # SAVE atómico — SIEMPRE GUARDA
    # =========================================================
    def _save_qvalues(self):
        path = self._json_path()
        dirpath = os.path.dirname(path)
        os.makedirs(dirpath, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=dirpath)
        with os.fdopen(fd, "w") as f:
            json.dump(self.Q, f)

        os.replace(tmp_path, path)
