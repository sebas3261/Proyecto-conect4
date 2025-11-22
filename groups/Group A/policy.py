import numpy as np
from connect4.policy import Policy
from typing import override


class Aha(Policy):
    def __init__(self):
        # RNG creado una sola vez
        self.rng = np.random.default_rng()
        self.fixed_col = None

    @override
    def mount(self) -> None:
        # Elegir columna fija solo una vez por partida
        self.fixed_col = int(self.rng.integers(0, 7))

    @override
    def act(self, s: np.ndarray) -> int:
        # Si la columna fija aún es válida
        if s[0, self.fixed_col] == 0:
            return self.fixed_col

        # Vectorizado: columnas donde la fila superior está vacía
        available_cols = np.flatnonzero(s[0] == 0)

        # Elegir nueva columna fija
        self.fixed_col = int(self.rng.choice(available_cols))
        return self.fixed_col

    def final(self, reward: int) -> None:
        pass