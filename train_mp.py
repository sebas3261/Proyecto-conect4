# ============================================================
#  TRAIN_MP.PY — Entrenamiento MULTICORE con PROMEDIO de Q
# ============================================================

import argparse
import multiprocessing
import numpy as np
import os
from collections import Counter, defaultdict

multiprocessing.freeze_support()

from connect4.policy import Policy
from connect4.utils import find_importable_classes
from connect4.connect_state import ConnectState


# ------------------------------------------------------------
# Partida entre dos políticas (1 vs -1)
# ------------------------------------------------------------
def play_single_game(name_plus, pol_plus, name_minus, pol_minus, rng) -> int:
    """Juega UNA partida entre dos policies ya instanciadas.
    Devuelve winner ∈ {1, -1, 0} desde la perspectiva del tablero (+1 / -1 / empate).
    """

    # Reiniciar memoria interna por partida
    pol_plus.mount()
    pol_minus.mount()

    state = ConnectState()

    while not state.is_final():
        # Usamos una copia para no mutar el tablero real en _normalize
        board = state.board.copy()

        if state.player == 1:
            action = pol_plus.act(board)
        else:
            action = pol_minus.act(board)

        state = state.transition(int(action))

    winner = state.get_winner()

    # Recompensas por jugador
    if winner == 1:
        pol_plus.final(+1)
        pol_minus.final(-1)
    elif winner == -1:
        pol_plus.final(-1)
        pol_minus.final(+1)
    else:
        pol_plus.final(0)
        pol_minus.final(0)

    return winner


# ------------------------------------------------------------
# Torneo knockout sencillo para "medir campeón" del worker
# ------------------------------------------------------------
def knockout_tournament(players: dict[str, Policy], rng) -> str:
    """Devuelve el nombre del campeón usando un bracket simple."""
    names = list(players.keys())
    rng.shuffle(names)

    while len(names) > 1:
        next_round = []
        for i in range(0, len(names), 2):
            if i + 1 == len(names):
                next_round.append(names[i])
                continue

            a = names[i]
            b = names[i + 1]
            pol_a = players[a]
            pol_b = players[b]

            # Tiramos moneda para ver quién juega como +1
            if rng.random() < 0.5:
                w = play_single_game(a, pol_a, b, pol_b, rng)
                if w == 1:
                    next_round.append(a)
                elif w == -1:
                    next_round.append(b)
                else:
                    next_round.append(rng.choice([a, b]))
            else:
                w = play_single_game(b, pol_b, a, pol_a, rng)
                if w == 1:
                    next_round.append(b)
                elif w == -1:
                    next_round.append(a)
                else:
                    next_round.append(rng.choice([a, b]))

        names = next_round

    return names[0]


# ------------------------------------------------------------
# Worker: entrena varias partidas y devuelve Q por grupo
# ------------------------------------------------------------
def worker_train(args):
    shuffle, seed, games_per_run = args
    rng = np.random.default_rng(seed)

    # Cada worker descubre sus grupos y crea UNA instancia por grupo
    participants = find_importable_classes("groups", Policy)  # {name: class}
    player_names = sorted(participants.keys())

    # Instancias persistentes dentro del worker
    players = {name: cls() for name, cls in participants.items()}

    # Jugar muchas partidas aleatorias entre grupos
    n_players = len(player_names)
    if n_players < 2:
        raise RuntimeError("Se necesitan al menos 2 grupos para entrenar.")

    for _ in range(games_per_run):
        # Elegimos dos grupos distintos al azar
        i, j = rng.choice(n_players, size=2, replace=False)
        name_a = player_names[i]
        name_b = player_names[j]
        pol_a = players[name_a]
        pol_b = players[name_b]

        # Aleatorio quién es +1 y quién -1
        if rng.random() < 0.5:
            play_single_game(name_a, pol_a, name_b, pol_b, rng)
        else:
            play_single_game(name_b, pol_b, name_a, pol_a, rng)

    # Torneo de evaluación para saber "campeón" de este worker
    champion_name = knockout_tournament(players, rng)

    # Extraer Q-values de cada policy que tenga atributo Q
    q_out: dict[str, dict] = {}
    for name, pol in players.items():
        if hasattr(pol, "Q"):
            q_out[name] = dict(pol.Q)  # copia ligera

    return champion_name, q_out


# ------------------------------------------------------------
# Fusionar Q-values de TODOS los workers usando PROMEDIO
# ------------------------------------------------------------
def merge_qvalues(all_q_out: list[dict[str, dict]]) -> dict[str, dict]:
    """
    all_q_out: lista (por worker) de dict[group_name -> {key -> q}]
    Devuelve dict[group_name -> {key -> q_promedio}]
    """
    sum_q: dict[str, dict] = {}
    count_q: dict[str, dict] = {}

    for worker_q in all_q_out:
        for group, qdict in worker_q.items():
            g_sum = sum_q.setdefault(group, {})
            g_cnt = count_q.setdefault(group, {})
            for key, val in qdict.items():
                g_sum[key] = g_sum.get(key, 0.0) + float(val)
                g_cnt[key] = g_cnt.get(key, 0) + 1

    final_q: dict[str, dict] = {}
    for group, g_sum in sum_q.items():
        g_cnt = count_q[group]
        final_q[group] = {k: g_sum[k] / g_cnt[k] for k in g_sum.keys()}

    return final_q


# ------------------------------------------------------------
# Guardar Q promedio en los archivos propios de cada policy
# ------------------------------------------------------------
def save_merged_qvalues(final_q: dict[str, dict]):
    """
    final_q: dict[group_name -> {key -> q}]
    Usa las propias policies para saber dónde guardar su JSON.
    """
    participants = find_importable_classes("groups", Policy)

    for name, cls in participants.items():
        if name not in final_q:
            continue

        pol = cls()
        if not hasattr(pol, "Q") or not hasattr(pol, "_save_qvalues"):
            continue

        pol.Q = final_q[name]
        pol._save_qvalues()

    print("\n🔥 Q-values promediados guardados en los JSON de cada grupo (B, C, D, ...)")


# ------------------------------------------------------------
# Entrenamiento paralelo de alto nivel
# ------------------------------------------------------------
def run_training_parallel(runs: int, shuffle: bool, seed: int, games_per_run: int):
    """
    runs          = cuántos 'jobs' lanzar (cada uno con games_per_run partidas)
    games_per_run = cuántas partidas juega cada worker
    """
    jobs = [(shuffle, seed + i, games_per_run) for i in range(runs)]

    ncpu = multiprocessing.cpu_count()
    print(f"🧵 Usando hasta {ncpu} núcleos para {runs} jobs...")
    print(f"   Cada job juega {games_per_run} partidas aleatorias entre grupos.\n")

    champions: list[str] = []
    all_q_out: list[dict[str, dict]] = []

    with multiprocessing.Pool(ncpu) as pool:
        for champion_name, q_out in pool.imap_unordered(worker_train, jobs):
            champions.append(champion_name)
            all_q_out.append(q_out)

    # Fusionar Q-values de todos los workers
    final_q = merge_qvalues(all_q_out)
    save_merged_qvalues(final_q)

    return champions


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento Connect4 MULTICORE con promedio de Q-values."
    )
    parser.add_argument("--runs", type=int, default=50,
                        help="Número de jobs (workers lógicos) a lanzar.")
    parser.add_argument("--games-per-run", type=int, default=200,
                        help="Partidas que juega cada job entre grupos.")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=911)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    champions = run_training_parallel(
        runs=args.runs,
        shuffle=args.shuffle,
        seed=args.seed,
        games_per_run=args.games_per_run,
    )

    # Estadísticas de campeones por job
    counter = Counter(champions)

    print("\n=== TRAINING FINISHED (MULTICORE + PROMEDIO) ===")
    print(f"Jobs ejecutados: {args.runs}")
    print("Frecuencia de campeones por job:")
    for name, cnt in counter.most_common():
        print(f"  {name}: {cnt}")
