# ==============================================================
#          TOURNAMENT.PY — MODO ULTRA TURBO (COPY/PASTE)
# ==============================================================

import numpy as np


# ==============================================================
# Emparejamientos iniciales (sin logging, ultra rápido)
# ==============================================================
def make_initial_matches(players, shuffle, rng):
    players = players[:]  # copiar para no mutar

    if shuffle:
        rng.shuffle(players)

    # siguiente potencia de 2
    def next_pow2(n):
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    size = next_pow2(len(players))

    # agregar BYEs (None)
    players += [None] * (size - len(players))

    return [(players[i], players[i + 1]) for i in range(0, size, 2)]


# ==============================================================
# Siguiente ronda
# ==============================================================
def pair_next_round(winners):
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


# ==============================================================
#        Versión ultra rápida de play()  — 1 partida
# ==============================================================

def play(a, b, seed=0):
    """
    Ultra-fast play function:
    - 1 single game
    - no JSON
    - no Match/Game objects
    - no Versus objects
    - calls .final() for learning
    - returns (name, policy_class) of the winner, or None
    """

    from connect4.connect_state import ConnectState

    rng = np.random.default_rng(seed)

    # Desempaquetar
    a_name, a_class = a
    b_name, b_class = b

    # Crear policies
    a_pol = a_class()
    b_pol = b_class()

    # Montar
    a_pol.mount()
    b_pol.mount()

    # Nuevo estado
    state = ConnectState()

    # Jugar hasta terminal
    while not state.is_final():
        board = state.board

        if state.player == 1:
            act = a_pol.act(board)
            state = state.transition_fast(int(act))
        else:
            act = b_pol.act(board)
            state = state.transition_fast(int(act))

    winner = state.get_winner()

    # Aprendizaje
    if winner == 1:
        a_pol.final(+1)
        b_pol.final(-1)
        return a
    elif winner == -1:
        a_pol.final(-1)
        b_pol.final(+1)
        return b
    else:
        a_pol.final(0)
        b_pol.final(0)
        return None


# ==============================================================
#           Torneo rápido (sin best-of, sin JSON)
# ==============================================================

def run_tournament(players, play_fn, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)

    # primera ronda
    versus = make_initial_matches(players, shuffle, rng)

    while True:
        winners = []

        for a, b in versus:
            # BYEs
            if a is None:
                winners.append(b)
                continue
            if b is None:
                winners.append(a)
                continue

            # jugar 1 única partida
            winner = play_fn(a, b, seed=seed)
            winners.append(winner)

        # si ya solo hay 1, terminó el torneo
        winners = [w for w in winners if w is not None]
        if len(winners) == 1:
            return winners[0]

        # emparejar para la siguiente ronda
        versus = pair_next_round(winners)
