import numpy as np
from connect4.policy import Policy
from typing import override


class Aha(Policy):

    @override
    def mount(self) -> None:
        rng = np.random.default_rng()
        self.fixed_col = int(rng.integers(0, 7)) 

    @override
    def act(self, s: np.ndarray) -> int:
        if s[0, self.fixed_col] == 0:
            return self.fixed_col

        # Si está llena, elegir cualquier columna válida
        available_cols = [c for c in range(7) if s[0, c] == 0]
        self.fixed_col = int(np.random.choice(available_cols))
        return self.fixed_col
