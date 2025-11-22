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


def find_winning_moves(board, cols, player):
    winning = []
    for c in cols:
        r = drop(board, c)
        if r is None:
            continue
        if board[r, c] != 0:
            continue
        board[r, c] = player
        if check_win_fast(board, r, c, player):
            winning.append(int(c))
        board[r, c] = 0
    return winning


class SuperDefensiveHeuristic(Policy):

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

        # 1. bloquear derrota inmediata primero (rival -1)
        opp_wins = find_winning_moves(board, cols, player=-1)
        if opp_wins:
            if 3 in opp_wins:
                return 3
            return opp_wins[0]

        # 2. luego, ganar si se puede (+1)
        my_wins = find_winning_moves(board, cols, player=1)
        if my_wins:
            if 3 in my_wins:
                return 3
            return my_wins[0]

        # 3. buscar jugadas relativamente "seguras": priorizar centro
        if 3 in cols:
            return 3

        # 4. columnas cercanas al centro
        for c in [2, 4, 1, 5, 0, 6]:
            if c in cols:
                return c

        return int(cols[0])
