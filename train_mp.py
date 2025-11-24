import argparse
import multiprocessing
import numpy as np
import os
import csv
import threading
from collections import Counter

# Agregar un lock global para asegurar acceso sincronizado a los Q-values
_qvalues_lock = threading.Lock()

multiprocessing.freeze_support()

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from connect4.connect_state import ConnectState


# ------------------------------------------------------------
# Partida entre dos policies (1 vs -1)
# ------------------------------------------------------------
def play_single_game(name_plus, pol_plus, name_minus, pol_minus, rng) -> tuple:
    """Devuelve:
    winner (1 / -1 / 0),
    total_moves,
    first_player (name_plus o name_minus)
    """

    pol_plus.mount()
    pol_minus.mount()

    state = ConnectState()
    moves = 0
    first_player = name_plus  # inicialmente quien usa +1

    while not state.is_final():
        board = state.board.copy()

        if state.player == 1:
            action = pol_plus.act(board)
        else:
            action = pol_minus.act(board)

        state = state.transition(int(action))
        moves += 1

    winner = state.get_winner()

    # Recompensas
    if winner == 1:
        pol_plus.final(+1)
        pol_minus.final(-1)
    elif winner == -1:
        pol_plus.final(-1)
        pol_minus.final(+1)
    else:
        pol_plus.final(0)
        pol_minus.final(0)

    return winner, moves, first_player


# ------------------------------------------------------------
# Torneo knockout (1 campeón por worker)
# ------------------------------------------------------------
def knockout_tournament(players: dict[str, Policy], rng) -> str:
    names = list(players.keys())
    rng.shuffle(names)

    while len(names) > 1:
        nxt = []
        for i in range(0, len(names), 2):
            if i + 1 >= len(names):
                nxt.append(names[i])
                continue

            a, b = names[i], names[i + 1]
            pa, pb = players[a], players[b]

            # aleatorio quién empieza
            if rng.random() < 0.5:
                w, _, _ = play_single_game(a, pa, b, pb, rng)
                nxt.append(a if w == 1 else b)
            else:
                w, _, _ = play_single_game(b, pb, a, pa, rng)
                nxt.append(b if w == 1 else a)

        names = nxt

    return names[0]


# ------------------------------------------------------------
# Worker: ENTRENAMIENTO + LOGGING + Q-values
# ------------------------------------------------------------
def worker_train(args):
    shuffle, seed, games_per_run, worker_id = args
    rng = np.random.default_rng(seed)

    # cargar policies por grupo
    participants = find_importable_classes("groups", Policy)
    player_names = list(sorted(participants.keys()))
    players = {name: cls() for name, cls in participants.items()}

    # LOG LOCAL DEL WORKER
    local_logs = []
    local_qvalues = {}  # Acumulador de Q-values por worker

    # entrenamiento aleatorio
    for _ in range(games_per_run):
        a, b = rng.choice(player_names, size=2, replace=False)
        pa, pb = players[a], players[b]

        # quién empieza
        if rng.random() < 0.5:
            winner, moves, fp = play_single_game(a, pa, b, pb, rng)
            fp_name = a
        else:
            winner, moves, fp = play_single_game(b, pb, a, pa, rng)
            fp_name = b

        # traducir ganador
        if winner == 1:
            win_name = fp_name
        elif winner == -1:
            win_name = b if fp_name == a else a
        else:
            win_name = "draw"

        # guardar fila
        local_logs.append({
            "worker": worker_id,
            "seed": seed,
            "player_a": a,
            "player_b": b,
            "first_player": fp_name,
            "winner": win_name,
            "moves": moves
        })

    # Acumular Q-values del worker
    for name, p in players.items():
        if hasattr(p, "Q"):
            local_qvalues[name] = dict(p.Q)

    # torneo final del worker
    champion = knockout_tournament(players, rng)

    return champion, local_qvalues, local_logs


# ------------------------------------------------------------
# Fusionar Q PROMEDIO
# ------------------------------------------------------------
def merge_qvalues(all_q_out):
    merged = {}
    counts = {}

    for w in all_q_out:
        for group, qdict in w.items():
            if group not in merged:
                merged[group] = {}
                counts[group] = {}

            for key, val in qdict.items():
                merged[group][key] = merged[group].get(key, 0.0) + val
                counts[group][key] = counts[group].get(key, 0) + 1

    final_q = {}
    for g in merged:
        final_q[g] = {k: merged[g][k] / counts[g][k] for k in merged[g]}

    return final_q


# ------------------------------------------------------------
# Guardar Q-values finales en cada policy
# ------------------------------------------------------------
def save_merged_qvalues(final_q):
    participants = find_importable_classes("groups", Policy)

    for name, cls in participants.items():
        if name not in final_q:
            continue
        p = cls()
        if hasattr(p, "Q") and hasattr(p, "_save_qvalues"):
            p.Q = final_q[name]
            p._save_qvalues()

    print("Q-values guardados correctamente.")


# ------------------------------------------------------------
# Entrenamiento MULTICORE
# ------------------------------------------------------------
def run_training_parallel(runs, shuffle, seed, games_per_run):
    jobs = [(shuffle, seed + i, games_per_run, i) for i in range(runs)]

    ncpu = multiprocessing.cpu_count()
    print(f"Usando {ncpu} núcleos para {runs} jobs…")

    champions = []
    all_q = []
    all_logs = []

    with multiprocessing.Pool(ncpu) as pool:
        for champion, q_out, logs in pool.imap_unordered(worker_train, jobs):
            champions.append(champion)
            all_q.append(q_out)
            all_logs.extend(logs)

    # Mezclar Q
    final_q = merge_qvalues(all_q)
    save_merged_qvalues(final_q)

    # ---- GUARDAR LOGS CSV ----
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/training_results.csv"

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_logs[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(all_logs)

    return champions


# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--games-per-run", type=int, default=200)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    champs = run_training_parallel(
        runs=args.runs,
        shuffle=args.shuffle,
        seed=args.seed,
        games_per_run=args.games_per_run
    )

    print("\n=== TRAINING FINISHED ===")
    print("Campeones por worker:")
    counter = Counter(champs)
    for name, cnt in counter.most_common():
        print(f"  {name}: {cnt}")
    print("Logs guardados en logs/training_results.csv")
