import numpy as np
from connect4.policy import Policy
from typing import override


def drop(board, col):
    col_data = board[:, col]
    for r in range(5, -1, -1):
        if col_data[r] == 0:
            return r
    return None


def check_win_fast(board, row, col, player):
    H, W = 6, 7

    # Horizontal
    count = 1
    for dc in (1, -1):
        c = col + dc
        while 0 <= c < W and board[row, c] == player:
            count += 1
            if count >= 4:
                return True
            c += dc

    # Vertical
    count = 1
    for dr in (1, -1):
        r = row + dr
        while 0 <= r < H and board[r, col] == player:
            count += 1
            if count >= 4:
                return True
            r += dr

    # Diagonal /
    count = 1
    for dr, dc in ((1, -1), (-1, 1)):
        r, c = row + dr, col + dc
        while 0 <= r < H and 0 <= c < W and board[r, c] == player:
            count += 1
            if count >= 4:
                return True
            r += dr
            c += dc

    # Diagonal \
    count = 1
    for dr, dc in ((1, 1), (-1, -1)):
        r, c = row + dr, col + dc
        while 0 <= r < H and 0 <= c < W and board[r, c] == player:
            count += 1
            if count >= 4:
                return True
            r += dr
            c += dc

    return False


class HybridHeuristic(Policy):

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

    def _score_move(self, board, col):
        r = drop(board, col)
        if r is None:
            return -1e9

        score = 0.0

        # peso posicional
        score += float(self.WEIGHTS[r, col])

        # ganar
        if board[r, col] != 0:
            return -1e9
        board[r, col] = 1
        if check_win_fast(board, r, col, 1):
            score += 1000.0
        board[r, col] = 0

        # bloquear rival
        board[r, col] = -1
        if check_win_fast(board, r, col, -1):
            score += 500.0
        board[r, col] = 0

        return score

    @override
    def act(self, board: np.ndarray) -> int:
        self._normalize(board)
        cols = np.flatnonzero(board[0] == 0)
        if cols.size == 0:
            return 0

        best_col = int(cols[0])
        best_val = -1e18

        for c in cols:
            v = self._score_move(board, int(c))
            if v > best_val:
                best_val = v
                best_col = int(c)

        return best_col
