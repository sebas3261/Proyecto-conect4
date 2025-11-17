import os
from typing import Callable
import numpy as np
from connect4.dtos import Game, Match, Participant, Versus, State, Action
from connect4.connect_state import ConnectState


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def make_initial_matches(players: list[Participant], shuffle: bool, seed: int) -> Versus:
    players = players[:]  # copy
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(players)

    size = next_power_of_two(len(players))
    players += [None] * (size - len(players))  # BYEs

    versus: Versus = []
    for i in range(0, size, 2):
        versus.append((players[i], players[i + 1]))
    return versus


def pair_next_round(winners: list[Participant]) -> Versus:
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def play_round(
    versus: Versus,
    play: Callable[[Participant, Participant, int, float, int], Participant],
    best_of: int,
    first_player_distribution: float,
    seed: int,
) -> list[Participant]:

    winners: list[Participant] = []

    for a, b in versus:
        if a is None and b is None:
            raise ValueError("Invalid match: two BYEs")
        if a is None:
            winners.append(b)
        elif b is None:
            winners.append(a)
        else:
            winners.append(play(a, b, best_of, first_player_distribution, seed))

    return winners


# ------------------------------------------------------------
# Jugar un MATCH (mejor de N)
# ------------------------------------------------------------
def play(
    a: Participant,
    b: Participant,
    best_of: int,
    first_player_distribution: float,
    seed: int = 911,
) -> Participant:

    a_name, a_policy = a
    b_name, b_policy = b

    a_wins = 0
    b_wins = 0
    draws = 0
    games_needed = (best_of // 2) + 1

    rng = np.random.default_rng(seed)
    games: list[Game] = []

    while a_wins < games_needed and b_wins < games_needed:

        # Elegir quién es +1 en esta partida
        if rng.random() < first_player_distribution:
            plus1_name, plus1 = a_name, a_policy()
            minus1_name, minus1 = b_name, b_policy()
        else:
            plus1_name, plus1 = b_name, b_policy()
            minus1_name, minus1 = a_name, a_policy()

        plus1.mount()
        minus1.mount()

        state = ConnectState()
        history: list[tuple[State, Action]] = []

        while not state.is_final():
            if state.player == 1:
                action = plus1.act(state.board)
            else:
                action = minus1.act(state.board)

            history.append((state.board.tolist(), int(action)))
            state = state.transition(int(action))

        # Guardar partida en formato nuevo
        games.append(
            Game(
                player_plus1=plus1_name,
                player_minus1=minus1_name,
                history=history,
            )
        )

        winner = state.get_winner()

        if winner == 1:     # +1 ganó
            if plus1_name == a_name:
                a_wins += 1
            else:
                b_wins += 1

        elif winner == -1:  # -1 ganó
            if minus1_name == a_name:
                a_wins += 1
            else:
                b_wins += 1

        else:
            draws += 1

        if draws > games_needed + 5:
            break

    # ------------------------------------------------------------
    # Guardar JSON (crear carpeta si no existe)
    # ------------------------------------------------------------
    os.makedirs("versus", exist_ok=True)

    match = Match(
        player_a=a_name,
        player_b=b_name,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        draws=draws,
        games=games,
    )

    with open(f"versus/match_{a_name}_vs_{b_name}.json", "w") as f:
        f.write(match.model_dump_json(indent=4))

    return a if a_wins > b_wins else b


# ------------------------------------------------------------
# Torneo completo
# ------------------------------------------------------------
def run_tournament(
    players: list[Participant],
    play: Callable[[Participant, Participant], Participant],
    best_of: int = 7,
    first_player_distribution: float = 0.5,
    shuffle: bool = True,
    seed: int = 911,
):
    versus = make_initial_matches(players, shuffle, seed)
    print("Initial Matches:", versus)

    while True:
        winners = play_round(
            versus, play, best_of, first_player_distribution, seed
        )
        print("Winners this round:", winners)

        if len(winners) == 1:
            return winners[0]

        versus = pair_next_round(winners)
        print("Next Matches:", versus)
