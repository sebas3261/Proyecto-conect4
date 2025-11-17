import json
import os
import re

# ================================
# Colores ANSI
# ================================
RED = "\033[91m"
BLUE = "\033[94m"
GRAY = "\033[90m"
RESET = "\033[0m"

# ================================
# Aplicar último movimiento
# ================================
def apply_move(board, col, player):
    """Pone la ficha final en su lugar (si falta)"""
    if col is None:
        return
    for r in reversed(range(6)):
        if board[r][col] == 0:
            board[r][col] = player
            return r
    return None

# ================================
# Detección de ganador
# ================================
def check_winner(board):
    H, W = 6, 7

    def ok(r, c):
        return 0 <= r < H and 0 <= c < W

    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for r in range(H):
        for c in range(W):
            if board[r][c] == 0:
                continue
            p = board[r][c]

            for dr, dc in dirs:
                if all(ok(r+dr*k, c+dc*k) and board[r+dr*k][c+dc*k] == p for k in range(4)):
                    return p
    return 0

# ================================
# Tablero con colores
# ================================
def print_board(board):
    for row in board:
        line = ""
        for x in row:
            if x == 1:
                line += BLUE + "● " + RESET
            elif x == -1:
                line += RED + "● " + RESET
            else:
                line += GRAY + "· " + RESET
        print(line)
    print()

# ================================
# Procesar UN archivo
# ================================
def process_file(path):
    print("\n===================================")
    print(f"📄 Archivo: {path}")
    print("===================================\n")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        print(RED + "❌ No se pudo leer el JSON\n" + RESET)
        return

    # Encabezado general
    print(f"Jugador A: {data['player_a']}")
    print(f"Jugador B: {data['player_b']}")
    print(f"Victorias A: {data['player_a_wins']}")
    print(f"Victorias B: {data['player_b_wins']}")
    print(f"Empates:    {data['draws']}\n")

    games = data["games"]

    print("=== DETALLE POR PARTIDA ===\n")

    for idx, game in enumerate(games, start=1):
        print(f"--- PARTIDA {idx} ---")

        # VALIDACIÓN DEL NUEVO FORMATO
        if not isinstance(game, dict) or "history" not in game:
            print(RED + "Partida vacía o corrupta\n" + RESET)
            continue

        history = game["history"]

        if not history:
            print(RED + "Partida vacía\n" + RESET)
            continue

        # Jugadores en esta partida
        plus1 = game.get("player_plus1", "Desconocido")
        minus1 = game.get("player_minus1", "Desconocido")

        # Extraer último movimiento
        last_state, last_col = history[-1]

        # Clonar tablero
        board = [row[:] for row in last_state]

        # Determinar qué jugador hizo ese movimiento
        turn = len(history)    # turno actual
        player = 1 if turn % 2 == 0 else -1

        # Aplicar movimiento final si falta
        apply_move(board, last_col, player)

        print(f"Última jugada: columna {last_col} (jugador {'+1' if player==1 else '-1'})")
        print(f"{('+1 = ' + plus1) if player==1 else ('-1 = ' + minus1)}\n")

        print("Tablero final:\n")
        print_board(board)

        winner = check_winner(board)

        if winner == 1:
            print("Ganador:", BLUE + f"{plus1} (+1)" + RESET)
        elif winner == -1:
            print("Ganador:", RED + f"{minus1} (-1)" + RESET)
        else:
            print("Resultado: Empate o sin 4 en línea")
        print()

# ================================
# EJECUCIÓN PRINCIPAL
# ================================
directory = "./versus"

pattern = re.compile(r"match_Group [A-H]_vs_Group [A-H]\.json$")

files = [f for f in os.listdir(directory) if pattern.match(f)]

if not files:
    print("❌ No se encontraron archivos en ./versus/")
else:
    for fname in files:
        process_file(os.path.join(directory, fname))
