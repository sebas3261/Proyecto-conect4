# ============================================================
#                   TRAIN.PY — ULTRA TURBO
# ============================================================

import argparse
import numpy as np
import os

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import make_initial_matches, pair_next_round, play as turbo_play


# ============================================================
# Torneo ultrarrápido con desempate automático
# ============================================================
def fast_run_tournament(players, shuffle, seed):
    rng = np.random.default_rng(seed)

    # Inicializar llaves del torneo
    versus = make_initial_matches(players, shuffle, rng)

    while True:
        winners = []

        for a, b in versus:

            # Avance automático si hay bye
            if a is None:
                winners.append(b)
                continue
            if b is None:
                winners.append(a)
                continue

            # Ejecutar un match rápido
            w = turbo_play(a, b, seed=seed)

            # Si hubo empate, repetir hasta que no haya
            while w is None:
                w = turbo_play(a, b, seed=int(seed + rng.integers(1e9)))

            winners.append(w)

        # Filtrar nulls
        winners = [w for w in winners if w is not None]

        # ¿Queda solo un campeón?
        if len(winners) == 1:
            return winners[0]

        # Siguiente ronda
        versus = pair_next_round(winners)


# ============================================================
# Guardar Q-values de TODAS las policies después del entrenamiento
# ============================================================
def save_all_qvalues(players):
    for name, policy_class in players:
        try:
            pol = policy_class()
            if hasattr(pol, "_save_qvalues"):
                pol._save_qvalues()
        except Exception as e:
            print(f"[WARN] No se pudo salvar Q de {name}: {e}")


# ============================================================
# Entrenamiento ultrarrápido
# ============================================================
def run_training(runs, best_of, fpd, shuffle, seed):
    # Buscar participantes
    participants = find_importable_classes("groups", Policy)
    players = list(participants.items())

    champions = []

    for i in range(runs):
        run_seed = seed + i

        champion = fast_run_tournament(
            players,
            shuffle,
            run_seed,
        )

        champions.append((i + 1, run_seed, champion[0]))

    # 🔥 Guardar Q de TODAS las policies al final
    save_all_qvalues(players)

    return champions, []


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-fast Connect4 training.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--first-player-distribution", type=float, default=0.5)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    champs, _ = run_training(
        runs=args.runs,
        best_of=args.best_of,
        fpd=args.first_player_distribution,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("\n=== TRAINING FINISHED (ULTRA TURBO MODE) ===")
    print("Torneos ejecutados:", args.runs)
    print("Campeones finales:")
    for run, seed, champ in champs:
        print(f"  Run {run} (seed={seed}) → Campeón: {champ}")
