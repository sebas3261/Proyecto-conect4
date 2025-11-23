import numpy as np
import json
import os
from connect4.policy import Policy
from typing import override


class EGreedyNormalized(Policy):

    def __init__(self):
        self.Q = {}                      # Q(s,a)
        self.alpha = 0.20                # learning rate
        self.epsilon = 0.05              # exploration prob
        self.rng = np.random.default_rng()

        self.memory = []                 # (state_key, action) for update

        self._load_qvalues()

    # ============================================================
    # MOUNT
    # ============================================================
    @override
    def mount(self, time_out=None):
        self.memory.clear()

    # ============================================================
    # NORMALIZAR — nunca muta el tablero original
    # ============================================================
    def _normalize(self, board: np.ndarray) -> np.ndarray:
        """Devuelve una versión normalizada del tablero sin mutarlo."""
        b = board.copy()
        ones = np.count_nonzero(b == 1)
        negs = np.count_nonzero(b == -1)

        # Si hay más -1, invertimos todo para que siempre seamos 1
        if negs > ones:
            b = -b

        return b

    # ============================================================
    # ACT
    # ============================================================
    @override
    def act(self, board: np.ndarray) -> int:

        # Copia real del tablero del entorno
        b_real = board.copy()

        # Tablero normalizado para el aprendizaje
        b = self._normalize(b_real)

        # Acciones posibles (basado en la posición REAL)
        available = np.flatnonzero(b_real[0] == 0)
        if available.size == 0:
            return 0  # columna cualquiera si está lleno

        # Clave del estado (tablero NORMALIZADO)
        state_key = b.tobytes().hex()

        # Inicializar Q-values para acciones nuevas
        for c in available:
            key = f"{state_key}|{c}"
            if key not in self.Q:
                self.Q[key] = 0.0

        # Epsilon-greedy
        if self.rng.random() < self.epsilon:
            action = int(self.rng.choice(available))
        else:
            action = max(available, key=lambda c: self.Q[f"{state_key}|{c}"])
            action = int(action)

        # Guardar paso en memoria para actualizar luego
        self.memory.append((state_key, action))

        return action

    # ============================================================
    # FINAL — Q-learning Monte Carlo
    # ============================================================
    @override
    def final(self, reward: int):

        # reward = +1 si ganó, -1 si perdió
        for state_key, action in self.memory:
            qkey = f"{state_key}|{action}"
            old = self.Q.get(qkey, 0.0)
            self.Q[qkey] = old + self.alpha * (reward - old)

        self.memory.clear()
        self._save_qvalues()

    # ============================================================
    # Q-values file utils
    # ============================================================

    def _json_path(self):
        """
        GUARDA SIEMPRE EN LA MISMA CARPETA que policy.py,
        de forma segura en cualquier OS o runner.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "qvalues.json")

    def _load_qvalues(self):
        path = self._json_path()

        if not os.path.exists(path):
            self.Q = {}
            return

        try:
            with open(path, "r") as f:
                txt = f.read().strip()
                self.Q = json.loads(txt) if txt else {}
        except Exception:
            # si está corrupto, empezar limpio
            self.Q = {}

    def _save_qvalues(self):
        """Guardado atómico y seguro."""
        path = self._json_path()
        tmp = path + ".tmp"

        with open(tmp, "w") as f:
            json.dump(self.Q, f)

        try:
            os.replace(tmp, path)
        except PermissionError:
            # Fix windows
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
