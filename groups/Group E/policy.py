import numpy as np
from connect4.policy import Policy
from typing import override


class ItsAMi(Policy):
    def __init__(self):
        # RNG persistente (MUCHO más rápido)
        self.rng = np.random.default_rng()

    @override
    def mount(self) -> None:
        # no necesitas nada aquí
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        # 🔥 Vectorizado: encontrar columnas válidas rápido
        available_cols = np.flatnonzero(s[0] == 0)

        # Seguridad, aunque no debería pasar
        if available_cols.size == 0:
            return 0

        # Elección aleatoria con RNG persistente
        return int(self.rng.choice(available_cols))
    
    def final(self, reward: int) -> None:
        pass
