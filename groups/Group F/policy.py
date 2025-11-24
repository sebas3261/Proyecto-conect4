import numpy as np
from connect4.policy import Policy
from typing import override

class BlockOrWinPolicy(Policy):
    @override
    def mount(self):
        # No guarda ningún estado
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        # Comprobar para cada columna si el jugador puede ganar o bloquear al oponente
        for col in range(7):
            if s[0, col] == 0:
                row = self._get_next_available_row(s, col)
                
                # Simular jugada del jugador (1)
                new_board = s.copy()  # Crea una copia del estado
                new_board[row, col] = 1
                if self._check_win(new_board, row, col, 1):
                    return col  # Ganamos, jugamos esta columna
                
                # Bloquear jugada del oponente (-1)
                new_board[row, col] = -1
                if self._check_win(new_board, row, col, -1):
                    return col  # Bloqueamos al oponente, jugamos esta columna

        # Si no hay jugada ganadora ni bloqueadora, elegir una columna aleatoria
        available_cols = [i for i in range(7) if s[0, i] == 0]
        return np.random.choice(available_cols)

    def _get_next_available_row(self, board: np.ndarray, col: int) -> int:
        """Devuelve la siguiente fila disponible en la columna"""
        for row in range(5, -1, -1):  # Desde la fila más baja
            if board[row, col] == 0:
                return row
        return -1

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Chequea si el jugador ha ganado colocando su ficha en (row, col)"""
        H, W = 6, 7
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal
        for dr, dc in directions:
            count = 1
            # Hacia adelante
            r, c = row + dr, col + dc
            while 0 <= r < H and 0 <= c < W and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Hacia atrás
            r, c = row - dr, col - dc
            while 0 <= r < H and 0 <= c < W and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False
    
    def final(self, reward):
        pass
