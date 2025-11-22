import argparse
import csv
import os
import json
from typing import Callable
from datetime import datetime

import numpy as np

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import make_initial_matches, pair_next_round
from tournament import play as default_play


# ==========================================================
# COLUMNAS PARA CADA CSV (evita columnas en blanco)
# ==========================================================
COLUMNS_MATCHES = [
    "timestamp",
    "run",
    "run_seed",
    "player_a",
    "player_b",
    "player_a_wins",
    "player_b_wins",
    "draws",
    "winner",
    "file",
    "best_of",
    "first_player_distribution",
    "shuffle",
]

COLUMNS_CHAMPIONS = [
    "timestamp",
    "run",
    "run_seed",
    "champion",
]

COLUMNS_STATS = [
    "timestamp",
    "run",
    "run_seed",
    "player",
    "matches_played",
    "matches_won",
    "matches_lost",
    "games_won",
    "games_lost",
    "games_drawn",
]


# ==========================================================
# CSV WRITER SEGURO (no sobrescribe columnas)
# ==========================================================
def write_csv(path: str, rows: list[dict], columns: list[str], append: bool = True):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)
    mode = "a" if (append and file_exists) else "w"

    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            clean_row = {col: row.get(col, "") for col in columns}
            writer.writerow(clean_row)


# ==========================================================
# MATCH LOGGING WRAPPER
# ==========================================================
def play_with_logging(
    a,
    b,
    best_of: int,
    first_player_distribution: float,
    seed: int,
    match_log: list[dict],
    inner_play: Callable,
):
    winner = inner_play(a, b, best_of, first_player_distribution, seed)

    a_name, _ = a
    b_name, _ = b

    match_filename = f"versus/match_{a_name}_vs_{b_name}.json"

    summary = {
        "player_a": a_name,
        "player_b": b_name,
        "run_seed": seed,
        "file": match_filename,
        "winner": winner[0] if isinstance(winner, tuple) else winner,
        "player_a_wins": 0,
        "player_b_wins": 0,
        "draws": 0,
    }

    try:
        with open(match_filename, "r") as f:
            data = json.load(f)
            summary["player_a_wins"] = data.get("player_a_wins", 0)
            summary["player_b_wins"] = data.get("player_b_wins", 0)
            summary["draws"] = data.get("draws", 0)
    except Exception:
        pass

    match_log.append(summary)
    return winner


# ==========================================================
# AGGREGATE PLAYER STATS
# ==========================================================
def build_player_stats(match_rows: list[dict]) -> list[dict]:
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
        a = row["player_a"]
        b = row["player_b"]

        a_wins = int(row.get("player_a_wins") or 0)
        b_wins = int(row.get("player_b_wins") or 0)
        draws = int(row.get("draws") or 0)
        winner = row.get("winner")

        for name in (a, b):
            s = ensure(name)
            s["matches_played"] += 1
            s["games_won"] += a_wins if name == a else b_wins
            s["games_lost"] += b_wins if name == a else a_wins
            s["games_drawn"] += draws

        if winner == a:
            ensure(a)["matches_won"] += 1
            ensure(b)["matches_lost"] += 1
        elif winner == b:
            ensure(b)["matches_won"] += 1
            ensure(a)["matches_lost"] += 1

    return list(stats.values())


# ==========================================================
# SAVE INDIVIDUAL RUN REPORT (NO SE SOBRESCRIBE)
# ==========================================================
def save_run_report(run_index: int, seed: int, matches: list[dict]):
    os.makedirs("versus/reports", exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = f"versus/reports/run_{run_index}_{seed}_{ts}.json"
    summary_path = f"versus/reports/run_{run_index}_{seed}_{ts}_summary.csv"

    stats = build_player_stats(matches)

    report_json = {
        "timestamp": datetime.now().isoformat(),
        "run": run_index,
        "seed": seed,
        "total_matches": len(matches),
        "total_games": sum(
            m.get("player_a_wins", 0) + m.get("player_b_wins", 0) + m.get("draws", 0)
            for m in matches
        ),
        "matches": matches,
        "player_stats": stats,
    }

    with open(report_path, "w") as f:
        json.dump(report_json, f, indent=2)

    write_csv(summary_path, stats, COLUMNS_STATS, append=False)


# ==========================================================
# TOURNAMENT ENGINE
# ==========================================================
def default_run_tournament(players, play_fn, best_of, fpd, shuffle, seed):
    versus = make_initial_matches(players, shuffle=shuffle, seed=seed)
    while True:
        winners = [_play_round_pair(p, play_fn, best_of, fpd, seed) for p in versus]
        winners = [w for w in winners if w is not None]
        if len(winners) == 1:
            return winners[0]
        versus = pair_next_round(winners)


def _play_round_pair(pair, play_fn, best_of, fpd, seed):
    a, b = pair
    if a is None:
        return b
    if b is None:
        return a
    return play_fn(a, b, best_of, fpd, seed)


# ==========================================================
# TRAINING LOOP — ACUMULATIVO Y SEGURO
# ==========================================================
def run_training(runs, best_of, fpd, shuffle, seed):
    participants = find_importable_classes("groups", Policy)
    players = list(participants.items())

    os.makedirs("versus", exist_ok=True)

    global_matches = []
    global_champions = []

    for i in range(runs):
        run_seed = seed + i
        match_log = []

        def wrapped_play(a, b, best_of_, fpd_, seed_=run_seed):
            return play_with_logging(a, b, best_of_, fpd_, seed_, match_log, default_play)

        champion = default_run_tournament(
            players,
            wrapped_play,
            best_of,
            fpd,
            shuffle,
            run_seed,
        )

        champion_name = champion[0] if isinstance(champion, tuple) else champion

        global_champions.append(
            {
                "timestamp": datetime.now().isoformat(),
                "run": i + 1,
                "run_seed": run_seed,
                "champion": champion_name,
            }
        )

        # annotate matches
        for row in match_log:
            row["timestamp"] = datetime.now().isoformat()
            row["run"] = i + 1
            row["run_seed"] = run_seed
            row["best_of"] = best_of
            row["first_player_distribution"] = fpd
            row["shuffle"] = shuffle
            global_matches.append(row)

        save_run_report(i + 1, run_seed, match_log)

    write_csv("versus/training_matches.csv", global_matches, COLUMNS_MATCHES, append=True)
    write_csv("versus/training_champions.csv", global_champions, COLUMNS_CHAMPIONS, append=True)
    write_csv(
        "versus/training_player_stats.csv",
        build_player_stats(global_matches),
        COLUMNS_STATS,
        append=True,
    )

    return global_matches, global_champions


# ==========================================================
# CLI
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple tournaments to train policies.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--best-of", type=int, default=7)
    parser.add_argument("--first-player-distribution", type=float, default=0.5)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    match_rows, champion_rows = run_training(
        runs=args.runs,
        best_of=args.best_of,
        fpd=args.first_player_distribution,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print(f"Completed {args.runs} tournaments.")
    print("Reports saved in versus/reports/")
