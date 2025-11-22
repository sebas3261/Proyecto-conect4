import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy


class UCB1Policy(Policy):

    def __init__(self):
        self.Q = {}         # Q-values
        self.N_s = {}       # visitas al estado
        self.N_sa = {}      # visitas a la acción
        self.memory = []    # historial (state_key, action)
        self.alpha = 0.2
        self.c = 2.0        # constante de exploración UCB
        self._load_qvalues()

    def mount(self, time_out=None):
        pass

    # ----------------------------
    # Selección de acción UCB1
    # ----------------------------
    def act(self, s: np.ndarray) -> int:
        s = s.copy()
        self._normalize(s)

        state_key = self._state_key(s)
        available_cols = [c for c in range(7) if s[0, c] == 0]

        # Incrementar visitas al estado
        self.N_s[state_key] = self.N_s.get(state_key, 0) + 1
        total_visits = self.N_s[state_key]

        best_value = -1e18
        best_action = available_cols[0]

        for a in available_cols:
            key_sa = f"{state_key}|{a}"

            # Crear Q inicial si no existe
            if key_sa not in self.Q:
                self.Q[key_sa] = 0.0

            q = self.Q[key_sa]
            n_sa = self.N_sa.get(key_sa, 0)

            # Fórmula UCB1
            bonus = self.c * np.sqrt(np.log(total_visits + 1) / (n_sa + 1))
            value = q + bonus

            if value > best_value:
                best_value = value
                best_action = a

        # Registrar transición
        self.memory.append((state_key, best_action))

        # Incrementar visitas s,a
        key = f"{state_key}|{best_action}"
        self.N_sa[key] = self.N_sa.get(key, 0) + 1

        # 🔥 Guardar inmediatamente (no esperar a final)
        self._save_qvalues()

        return best_action

    # ----------------------------
    # Actualización al final
    # ----------------------------
    def final(self, reward):
        for state_key, action in self.memory:
            key = f"{state_key}|{action}"
            old_q = self.Q.get(key, 0.0)
            new_q = old_q + self.alpha * (reward - old_q)
            self.Q[key] = new_q

        self.memory = []
        self._save_qvalues()

    # ----------------------------
    # Normalización
    # ----------------------------
    def _normalize(self, s):
        ones = np.sum(s == 1)
        negs = np.sum(s == -1)
        if negs == ones + 1:
            s[:] = -s

    def _state_key(self, s):
        return ",".join(map(str, s.reshape(-1)))

    # ----------------------------
    # Persistencia robusta
    # ----------------------------
    def _json_path(self):
        return os.path.join(os.path.dirname(__file__), "qvalues_ucb.json")

    def _load_qvalues(self):
        path = self._json_path()
        if not os.path.exists(path):
            self.Q = {}
            return
        try:
            with open(path, "r") as f:
                self.Q = json.load(f)
        except:
            self.Q = {}

    def _save_qvalues(self):
        path = self._json_path()

        # 🔥 Ahora SIEMPRE guarda aunque Q sea vacío
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            json.dump(self.Q, f)

        os.replace(tmp_path, path)
