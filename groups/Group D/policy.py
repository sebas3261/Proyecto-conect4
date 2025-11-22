import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class TrialBasedOPIPolicy(Policy):

    def __init__(self):
        self.Q = {}
        self.alpha = 0.15
        self.rollout_depth = 3   # profundidad limitada
        self.memory = []
        self._load_qvalues()

    def mount(self, time_out=None):
        pass

    # -------------------------------------------
    # Selección usando TB-OPI (rollouts)
    # -------------------------------------------
    def act(self, s: np.ndarray) -> int:
        s = s.copy()
        self._normalize(s)

        state_key = self._state_key(s)
        available_cols = [c for c in range(7) if s[0, c] == 0]

        best_val = -1e18
        best_action = available_cols[0]

        for a in available_cols:
            value = self._estimate_action_value(s, a)

            if value > best_val:
                best_val = value
                best_action = a

        # Guardar transición para update on-policy
        self.memory.append((state_key, best_action))

        # On-policy update
        key = f"{state_key}|{best_action}"
        old_q = self.Q.get(key, 0.0)
        new_q = old_q + self.alpha * (best_val - old_q)
        self.Q[key] = new_q

        self._save_qvalues()

        return best_action

    # -------------------------------------------
    # Rollout aleatorio corto
    # -------------------------------------------
    def _estimate_action_value(self, s, action):
        try:
            sim = ConnectState(s.copy())
        except:
            sim = s.copy()

        reward = self._simulate(sim, action, depth=self.rollout_depth)
        return reward

    def _simulate(self, state, action, depth):
        try:
            next_s, reward, done = state.transition(action)
        except:
            return 0

        if done or depth == 0:
            return reward if reward is not None else 0

        # Elegir una acción aleatoria válida
        available = [c for c in range(7) if next_s[0, c] == 0]
        if not available:
            return 0

        a = np.random.choice(available)
        return self._simulate(ConnectState(next_s.copy()), a, depth - 1)

    # -------------------------------------------
    # final() no hace nada grande (ya actualizamos en act)
    # -------------------------------------------
    def final(self, reward):
        pass

    # -------------------------------------------
    # Utilidades
    # -------------------------------------------
    def _normalize(self, s):
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)
        if negs == ones + 1:
            s[:] = -s

    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues_opi.json")

    def _load_qvalues(self):
        path = self._json_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                self.Q = json.load(f)
        except:
            self.Q = {}

    def _save_qvalues(self):
        if len(self.Q) == 0:
            return
        path = self._json_path()
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            json.dump(self.Q, f)
        os.replace(tmp_path, path)
