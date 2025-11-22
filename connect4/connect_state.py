from connect4.environment_state import EnvironmentState
import numpy as np
from typing import List


class ConnectState(EnvironmentState):
    ROWS = 6
    COLS = 7

    # (LINES se llenará después de la clase)

    # ------------------------------------------------------
    def __init__(self, board: np.ndarray | None = None, player: int = -1):
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        else:
            self.board = board.astype(np.int8)

        self.player = int(player)

        # Alturas O(1)
        self.heights = np.zeros(self.COLS, dtype=np.int8)
        for c in range(self.COLS):
            col = self.board[:, c]
            nz = np.nonzero(col)[0]
            if nz.size == 0:
                self.heights[c] = 0
            else:
                self.heights[c] = self.ROWS - nz[0]

        # Espacios libres
        self.empty_count = int(np.count_nonzero(self.board == 0))

        # Cache del ganador
        self._winner = 0

    # ------------------------------------------------------
    def _check_after_move(self, row: int, col: int) -> int:
        player = self.board[row, col]
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

        r = c = 0  # evita warnings

        for dr, dc in dirs:
            count = 1

            # forward
            r, c = row + dr, col + dc
            while (
                0 <= r < self.ROWS and
                0 <= c < self.COLS and
                self.board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc

            # backward
            r, c = row - dr, col - dc
            while (
                0 <= r < self.ROWS and
                0 <= c < self.COLS and
                self.board[r, c] == player
            ):
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return player

        return 0

    # ------------------------------------------------------
    def transition_fast(self, col: int):
        if self.board[0, col] != 0:
            raise ValueError(f"Column {col} is full.")

        h = self.heights[col]
        row = self.ROWS - 1 - h

        self.board[row, col] = self.player
        self.heights[col] += 1
        self.empty_count -= 1

        self._winner = self._check_after_move(row, col)

        self.player = -self.player
        return self

    # ------------------------------------------------------
    def transition(self, col: int):
        new = ConnectState(self.board.copy(), self.player)
        return new.transition_fast(col)

    # ------------------------------------------------------
    def is_final(self) -> bool:
        return self._winner != 0 or self.empty_count == 0

    def get_winner(self) -> int:
        return self._winner

    # ------------------------------------------------------
    def is_applicable(self, col: int) -> bool:
        return (
            0 <= col < self.COLS
            and self.board[0, col] == 0
            and not self.is_final()
        )

    # ------------------------------------------------------
    def get_free_cols(self) -> List[int]:
        return np.flatnonzero(self.board[0] == 0).tolist()

    # ------------------------------------------------------
    def get_heights(self) -> List[int]:
        return self.heights.tolist()



# ------------------------------------------------------
# PRECÁLCULO FINAL DE LÍNEAS GANADORAS (FUERA DE LA CLASE)
# ------------------------------------------------------

def _compute_lines():
    lines = []
    ROWS = ConnectState.ROWS
    COLS = ConnectState.COLS

    for r in range(ROWS):
        for c in range(COLS):

            # Horizontal →
            if c + 3 < COLS:
                lines.append([(r, c + i) for i in range(4)])

            # Vertical ↓
            if r + 3 < ROWS:
                lines.append([(r + i, c) for i in range(4)])

            # Diagonal ↘
            if r + 3 < ROWS and c + 3 < COLS:
                lines.append([(r + i, c + i) for i in range(4)])

            # Diagonal ↙
            if r + 3 < ROWS and c - 3 >= 0:
                lines.append([(r + i, c - i) for i in range(4)])

    return np.array(lines, dtype=np.int8)


# Asignar el resultado a la clase (ya existe aquí)
ConnectState.LINES = _compute_lines()
