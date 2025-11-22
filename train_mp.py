# ============================================================
#    TRAIN_MP.PY — Entrenamiento MULTICORE con FUSIÓN Q-values
# ============================================================

import argparse
import numpy as np
import multiprocessing
import json
import os

multiprocessing.freeze_support()

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import make_initial_matches, pair_next_round


# ------------------------------------------------------------
# Worker: ejecuta 1 torneo y guarda sus Q-values
# ------------------------------------------------------------
def fast_run_tournament(args):
    from tournament import play as turbo_play  # Safe-import en Windows

    players, shuffle, seed, worker_id = args

    rng = np.random.default_rng(seed)
    versus = make_initial_matches(players, shuffle, rng)

    # Jugar torneo
    while True:
        winners = []
        for a, b in versus:
            if a is None:
                winners.append(b)
                continue
            if b is None:
                winners.append(a)
                continue

            w = turbo_play(a, b, seed=seed)

            while w is None:
                w = turbo_play(a, b, seed=seed + rng.integers(1e9))

            winners.append(w)

        winners = [w for w in winners if w is not None]

        if len(winners) == 1:
            champion = winners[0]
            break

        versus = pair_next_round(winners)

    # --------------------------------------------------------
    # Guardar Q-values de cada policy del worker
    # --------------------------------------------------------
    out = {}

    for name, policy_class in players:
        policy = policy_class()
        if hasattr(policy, "Q"):
            out[name] = policy.Q

    # Guardar JSON del worker
    worker_file = f"qvalues_worker{worker_id}.json"
    with open(worker_file, "w") as f:
        json.dump(out, f)

    return champion


# ------------------------------------------------------------
# Fusionar todos los JSON en uno solo
# ------------------------------------------------------------
def merge_qvalues(num_workers):
    final_q = {}

    for i in range(num_workers):
        fname = f"qvalues_worker{i}.json"
        if not os.path.exists(fname):
            continue

        try:
            with open(fname, "r") as f:
                data = json.load(f)
        except:
            continue

        # Fusionar Q-values por grupo
        for group, qdict in data.items():
            if group not in final_q:
                final_q[group] = {}

            for key, val in qdict.items():
                # Estrategia: promedio
                if key not in final_q[group]:
                    final_q[group][key] = val
                else:
                    final_q[group][key] = (final_q[group][key] + val) / 2

    # Guardar archivo final
    with open("qvalues_final.json", "w") as f:
        json.dump(final_q, f)

    print("\n🔥 Q-values fusionados en: qvalues_final.json")
    print("🔥 Puedes seguir entrenando usando este archivo\n")


# ------------------------------------------------------------
# Entrenamiento paralelo
# ------------------------------------------------------------
def run_training_parallel(runs, shuffle, seed):
    participants = find_importable_classes("groups", Policy)
    players = list(participants.items())

    jobs = [(players, shuffle, seed + i, i) for i in range(runs)]

    ncpu = multiprocessing.cpu_count()
    print(f"🧵 Usando {ncpu} núcleos para {runs} torneos...")

    with multiprocessing.Pool(ncpu) as pool:
        champions = pool.map(fast_run_tournament, jobs)

    merge_qvalues(runs)

    return champions


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    champs = run_training_parallel(
        runs=args.runs,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("\n=== TRAINING FINISHED (MULTICORE MODE + Q MERGE) ===")
    print("Torneos ejecutados:", args.runs)
    for i, champ in enumerate(champs):
        print(f"  Run {i+1}: {champ[0]}")
