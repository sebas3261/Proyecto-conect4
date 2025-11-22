import argparse
import csv
import os
from typing import Callable

import numpy as np

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import make_initial_matches, pair_next_round
from tournament import play as default_play


def play_with_logging(
    a,
    b,
    best_of: int,
    first_player_distribution: float,
    seed: int,
    match_log: list[dict],
    inner_play: Callable,
):
    """
    Wraps the default play to capture match stats without changing training behaviour.
    """
    winner = inner_play(a, b, best_of, first_player_distribution, seed)

    # default_play already writes match_*.json in versus/, reuse the same seed so names match
    # We log just the summary (wins/draws) by reading the generated file.
    a_name, _ = a
    b_name, _ = b
    match_filename = f"versus/match_{a_name}_vs_{b_name}.json"
    summary = {
        "run_seed": seed,
        "player_a": a_name,
        "player_b": b_name,
        "player_a_wins": None,
        "player_b_wins": None,
        "draws": None,
        "winner": winner[0] if isinstance(winner, tuple) else winner,
        "file": match_filename,
    }
    # Optionally parse file if it exists (best-effort)
    try:
        import json

        with open(match_filename, "r") as f:
            data = json.load(f)
            summary["player_a_wins"] = data.get("player_a_wins")
            summary["player_b_wins"] = data.get("player_b_wins")
            summary["draws"] = data.get("draws")
    except FileNotFoundError:
        pass

    match_log.append(summary)
    return winner


def run_training(
    runs: int,
    best_of: int,
    first_player_distribution: float,
    shuffle: bool,
    seed: int,
):
    participants = find_importable_classes("groups", Policy)
    players = list(participants.items())

    os.makedirs("versus", exist_ok=True)

    match_rows: list[dict] = []
    champion_rows: list[dict] = []

    for i in range(runs):
        run_seed = seed + i
        match_log: list[dict] = []

        def wrapped_play(a, b, best_of_, fpd_, seed_=run_seed):
            # Use a per-run seed so tournaments differ
            return play_with_logging(a, b, best_of_, fpd_, seed_, match_log, default_play)

        champion = default_run_tournament(
            players,
            wrapped_play,
            best_of=best_of,
            first_player_distribution=first_player_distribution,
            shuffle=shuffle,
            seed=run_seed,
        )

        champion_rows.append(
            {
                "run": i + 1,
                "seed": run_seed,
                "champion": champion[0] if isinstance(champion, tuple) else champion,
            }
        )
        for row in match_log:
            row["run"] = i + 1
            row["best_of"] = best_of
            row["first_player_distribution"] = first_player_distribution
            row["shuffle"] = shuffle
            match_rows.append(row)

    # Write CSVs
    write_csv("versus/training_matches.csv", match_rows)
    write_csv("versus/training_champions.csv", champion_rows)
    write_csv("versus/training_player_stats.csv", build_player_stats(match_rows))

    return match_rows, champion_rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_player_stats(match_rows: list[dict]) -> list[dict]:
    """
    Aggregate per-player stats across all matches. Helpful for quick plotting.
    """
    stats: dict[str, dict] = {}

    def ensure(name: str):
        if name not in stats:
            stats[name] = {
                "player": name,
                "matches_played": 0,
                "matches_won": 0,
                "matches_lost": 0,
                "games_won": 0,
                "games_lost": 0,
                "games_drawn": 0,
            }
        return stats[name]

    for row in match_rows:
        a = row.get("player_a")
        b = row.get("player_b")
        a_wins = int(row.get("player_a_wins") or 0)
        b_wins = int(row.get("player_b_wins") or 0)
        draws = int(row.get("draws") or 0)
        winner = row.get("winner")

        for name in (a, b):
            if name is None:
                continue
            s = ensure(name)
            s["matches_played"] += 1
            s["games_won"] += a_wins if name == a else b_wins
            s["games_lost"] += b_wins if name == a else a_wins
            s["games_drawn"] += draws

        if winner and winner == a:
            ensure(a)["matches_won"] += 1
            ensure(b)["matches_lost"] += 1
        elif winner and winner == b:
            ensure(b)["matches_won"] += 1
            ensure(a)["matches_lost"] += 1

    return list(stats.values())


# Local copy of run_tournament so we can inject our play wrapper
def default_run_tournament(
    players: list,
    play_fn: Callable,
    best_of: int,
    first_player_distribution: float,
    shuffle: bool,
    seed: int,
):
    versus = make_initial_matches(players, shuffle=shuffle, seed=seed)
    while True:
        winners = []
        for pair in versus:
            winners.append(
                _play_round_pair(pair, play_fn, best_of, first_player_distribution, seed)
            )
        winners = [w for w in winners if w is not None]
        if len(winners) == 1:
            return winners[0]
        versus = pair_next_round(winners)


def _play_round_pair(pair, play_fn, best_of, fpd, seed):
    a, b = pair
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return play_fn(a, b, best_of, fpd, seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple tournaments to train policies.")
    parser.add_argument("--runs", type=int, default=10, help="Number of tournaments to run.")
    parser.add_argument("--best-of", type=int, default=7, help="Games per match.")
    parser.add_argument(
        "--first-player-distribution",
        type=float,
        default=0.5,
        help="Probability the first listed participant starts.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle initial pairings.",
    )
    parser.add_argument("--seed", type=int, default=911, help="Base random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    match_rows, champion_rows = run_training(
        runs=args.runs,
        best_of=args.best_of,
        first_player_distribution=args.first_player_distribution,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Completed {args.runs} tournaments.")
    print(f"Match log saved to versus/training_matches.csv ({len(match_rows)} rows).")
    print(f"Champion log saved to versus/training_champions.csv ({len(champion_rows)} rows).")
