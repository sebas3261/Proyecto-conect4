from typing import Callable
from connect4.dtos import Game, Match, Participant, Versus
from connect4.connect_state import ConnectState
import numpy as np
from connect4.dtos import State, Action



def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def make_initial_matches(
    players: list[Participant], shuffle: bool, seed: int
) -> Versus:
    """Create the first round, padding with BYEs (None) up to a power of two."""
    players = players[:]  # copy
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(players)
    size = next_power_of_two(len(players))
    players += [None] * (size - len(players))  # BYEs
    return [(players[i], players[i + 1]) for i in range(0, len(players), 2)]


def play_round(
    versus: Versus,
    play: Callable[[Participant, Participant, int, float, int], Participant],
    best_of: int,
    first_player_distribution: float,
    seed: int,
) -> list[Participant]:
    """Run a round and return the list of winners (handles BYEs)."""
    winners: list[Participant] = []
    for a, b in versus:
        if a is None and b is None:
            raise ValueError("Invalid match: two BYEs")
        if a is None:  # b advances
            winners.append(b)
        elif b is None:  # a advances
            winners.append(a)
        else:
            winners.append(play(a, b, best_of, first_player_distribution, seed))
    return winners


def pair_next_round(winners: list[Participant]) -> Versus:
    """Pair adjacent winners for the next round."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


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
    total_games = 0
    games_to_win = (best_of // 2) + 1

    rng = np.random.default_rng(seed)

    games: list[Game] = []

    while a_wins < games_to_win and b_wins < games_to_win:

        total_games += 1

        # Decide quién es +1
        if rng.random() < first_player_distribution:
            plus1_name, plus1_policy = a_name, a_policy()
            minus1_name, minus1_policy = b_name, b_policy()
        else:
            plus1_name, plus1_policy = b_name, b_policy()
            minus1_name, minus1_policy = a_name, a_policy()

        # Inicializar agente
        plus1_policy.mount()
        minus1_policy.mount()

        state = ConnectState()
        history: list[tuple[State, Action]] = []

        # Jugar
        while not state.is_final():
            if state.player == 1:
                action = plus1_policy.act(state.board)
            else:
                action = minus1_policy.act(state.board)

            history.append((state.board.tolist(), int(action)))
            state = state.transition(int(action))

        # Guardar partida
        games.append(Game(
            player_plus1 = plus1_name,
            player_minus1 = minus1_name,
            history = history
        ))

        # Determinar ganador
        winner = state.get_winner()

        if winner == 1:        # ganador = +1
            if plus1_name == a_name:
                a_wins += 1
            else:
                b_wins += 1
        elif winner == -1:     # ganador = -1
            if minus1_name == a_name:
                a_wins += 1
            else:
                b_wins += 1
        else:
            draws += 1

        if draws >= games_to_win + 5:
            break

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





def run_tournament(
    players: list[Participant],
    play: Callable[[Participant, Participant], Participant],
    best_of: int = 7,
    first_player_distribution: float = 0.5,
    shuffle: bool = True,
    seed: int = 911,
):
    """
    Run a tournament among the given players using the provided play function.

    Parameters
    ----------
    players : List[Participant]
        List of participants (name, policy) tuples.
    play : Callable[[Participant, Participant], Participant]
        Function that takes two participants and returns the winner.
    best_of : int, optional
        Number of games per match (default is 7).
    first_player_distribution : float, optional
        Distribution of games as first player (default is 0.5).
    shuffle : bool, optional
        Whether to shuffle initial pairings (default is True).
    seed : int, optional
        Random seed for reproducibility (default is 911).

    """
    versus = make_initial_matches(players, shuffle=shuffle, seed=seed)
    print("Initial Matches:", versus)
    while True:
        winners = play_round(versus, play, best_of, first_player_distribution, seed)
        print("Winners this round:", winners)
        if len(winners) == 1:  # champion decided
            return winners[0]
        versus = pair_next_round(winners)
        print("Next Matches:", versus)
