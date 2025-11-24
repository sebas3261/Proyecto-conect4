import numpy as np
from connect4.policy import Policy
from typing import override

class CenterDominationPolicy(Policy):
    @override
    def mount(self):
        # No guarda ningún estado
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        # Prioriza el centro y las columnas cercanas
        central_cols = [3]  # Columna central
        nearby_cols = [2, 4]  # Columnas cercanas al centro

        # Primero intentar jugar en el centro
        for col in central_cols:
            if s[0, col] == 0:
                return col
        
        # Luego en las columnas cercanas al centro
        for col in nearby_cols:
            if s[0, col] == 0:
                return col

        # Si ninguna de las anteriores opciones, elige una columna aleatoria
        available_cols = [i for i in range(7) if s[0, i] == 0]
        return np.random.choice(available_cols)
    
    def final(self, reward):
        pass
