import numpy as np
from connect4.policy import Policy
from typing import override


def drop(board, col):
    col_data = board[:, col]
    for r in range(5, -1, -1):
        if col_data[r] == 0:
            return r
    return None


class PositionalHeuristic(Policy):

    WEIGHTS = np.array([
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3],
    ])

    def __init__(self):
        self.rng = np.random.default_rng()

    @override
    def mount(self, time_out=None):
        pass

    def _normalize(self, board):
        ones = np.count_nonzero(board == 1)
        negs = np.count_nonzero(board == -1)
        if negs > ones:
            board *= -1

    @override
    def act(self, board: np.ndarray) -> int:
        self._normalize(board)
        cols = np.flatnonzero(board[0] == 0)
        if cols.size == 0:
            return 0

        best_col = int(cols[0])
        best_val = -1e18

        for c in cols:
            r = drop(board, int(c))
            if r is None:
                continue
            v = float(self.WEIGHTS[r, int(c)])
            if v > best_val:
                best_val = v
                best_col = int(c)

        return best_col
