import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy
from typing import override


class UncertaintyWithEGreedy(Policy):

    def __init__(self):
        self.Q = {}
        self.memory = []
        self.epsilon = 0.01
        self.alpha = 0.2
        self.rng = np.random.default_rng()
        self._load_qvalues()

    @override
    def mount(self, time_out=None):
        self.memory.clear()

    @override
    def act(self, s: np.ndarray) -> int:
        self._normalize(s)

        state_key = self._state_key(s)
        cols = np.flatnonzero(s[0] == 0)
        if cols.size == 0:
            return 0

        for c in cols:
            key = f"{state_key}|{int(c)}"
            if key not in self.Q:
                self.Q[key] = 0.0

        # e-greedy
        if self.rng.random() < self.epsilon:
            a = int(self.rng.choice(cols))
        else:
            a = max(cols, key=lambda x: self.Q.get(f"{state_key}|{int(x)}", 0.0))

        self.memory.append((state_key, a))
        return a

    @override
    def final(self, reward):
        for s_key, a in self.memory:
            key = f"{s_key}|{a}"
            q = self.Q.get(key, 0.0)
            self.Q[key] = q + self.alpha * (reward - q)
        self.memory.clear()

    # utils ------------

    def _normalize(self, s):
        if np.count_nonzero(s == -1) > np.count_nonzero(s == 1):
            s *= -1

    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues.json")

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

    # MODERNIZADO (multicore safe)
    def _save_qvalues(self, path_override=None):
        path = path_override or self._json_path()
        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(self.Q, f)

        try:
            os.replace(tmp, path)
        except PermissionError:
            import time
            time.sleep(0.05)
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
