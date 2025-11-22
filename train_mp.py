# ============================================================
#   TRAIN_MP.PY — MULTICORE TRAINING + CSV LOGGING + Q-MERGE
# ============================================================

import argparse
import numpy as np
import multiprocessing
import json
import os
import csv
import time
multiprocessing.freeze_support()

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import make_initial_matches, pair_next_round, play


# ------------------------------------------------------------
# Ejecuta un torneo completo en un proceso
# ------------------------------------------------------------
def fast_run_tournament(args):
    players, shuffle, seed, worker_id = args

    rng = np.random.default_rng(seed)
    versus = make_initial_matches(players, shuffle, rng)

    # Log por run
    log = {
        "run": worker_id,
        "seed": seed,
        "winner": None,
        "loser": None,
        "winner_policy": None,
        "loser_policy": None,
        "total_moves": 0,
        "duration": 0.0,
        "first_player": None,
        "fp_advantage": None,
        "exploration_rate": None,
        "reward": None,
        "win_type": None,
    }

    start_time = time.time()

    while True:
        winners = []
        for a, b in versus:
            if a is None:
                winners.append(b)
                continue
            if b is None:
                winners.append(a)
                continue

            # PARTIDA ÚNICA (best-of = 1 para máximo rendimiento)
            result = play(a, b, best_of=1, first_player_distribution=0.5, seed=seed)

            if result is None:
                continue

            winner_name, loser_name, match_data = result

            winners.append((winner_name, players_dict[winner_name]))

            # ---- Guardar información del match ----
            log["winner"] = winner_name
            log["loser"] = loser_name
            log["winner_policy"] = players_dict[winner_name].__name__
            log["loser_policy"] = players_dict[loser_name].__name__
            log["total_moves"] = match_data["moves"]
            log["first_player"] = match_data["first_player"]
            log["fp_advantage"] = 1 if match_data["first_player"] == winner_name else 0
            log["reward"] = match_data["reward"]
            log["win_type"] = match_data["result"]

        if len(winners) == 1:
            log["winner"] = winners[0][0]
            break

        versus = pair_next_round(winners)

    end_time = time.time()
    log["duration"] = round(end_time - start_time, 5)

    # Guardar Q-values del worker
    q_out = {}
    for name, cls in players:
        try:
            p = cls()
            if hasattr(p, "Q"):
                q_out[name] = p.Q
        except:
            pass

    with open(f"qvalues_worker_{worker_id}.json", "w") as f:
        json.dump(q_out, f)

    return log


# ------------------------------------------------------------
# Fusionar Q-values de todos los workers
# ------------------------------------------------------------
def merge_qvalues(num_workers):
    merged = {}

    for i in range(num_workers):
        path = f"qvalues_worker_{i}.json"
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            data = json.load(f)

        for group, qdict in data.items():
            if group not in merged:
                merged[group] = {}

            for key, val in qdict.items():
                if key not in merged[group]:
                    merged[group][key] = val
                else:
                    merged[group][key] = (merged[group][key] + val) / 2

    with open("qvalues_final.json", "w") as f:
        json.dump(merged, f, indent=2)

    print("🔥 Q-values fusionados → qvalues_final.json")


# ------------------------------------------------------------
# Training en paralelo
# ------------------------------------------------------------
def run_training_parallel(runs, shuffle, seed):
    global players_dict

    participants = find_importable_classes("groups", Policy)
    players = list(participants.items())
    players_dict = {name: cls for name, cls in players}

    jobs = [(players, shuffle, seed + i, i) for i in range(runs)]

    ncpu = multiprocessing.cpu_count()
    print(f"🧵 Usando {ncpu} núcleos para {runs} torneos...")

    with multiprocessing.Pool(ncpu) as pool:
        logs = pool.map(fast_run_tournament, jobs)

    # Guardar CSV
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/training_results.csv"

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(logs)

    merge_qvalues(runs)

    return logs


# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logs = run_training_parallel(args.runs, args.shuffle, args.seed)

    print("\n=== TRAINING FINISHED ===")
    print("Logs CSV guardados en logs/training_results.csv")
    print("Q-values guardados en qvalues_final.json")
