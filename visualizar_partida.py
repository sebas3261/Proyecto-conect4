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
    for row in reversed(range(6)):
        if board[row][col] == 0:
            board[row][col] = player
            return row
    return None

# ================================
# Detección de ganador
# ================================
def check_winner(board):
    H, W = 6, 7

    def valid(r, c):
        return 0 <= r < H and 0 <= c < W

    dirs = [(1,0), (0,1), (1,1), (1,-1)]

    for r in range(H):
        for c in range(W):
            if board[r][c] == 0:
                continue
            player = board[r][c]
            for dr, dc in dirs:
                if all(valid(r+dr*k, c+dc*k) and board[r+dr*k][c+dc*k] == player for k in range(4)):
                    return player
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
    print(f"\n==============================")
    print(f"📄 Archivo: {path}")
    print(f"==============================")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        print(RED + "❌ Error leyendo JSON" + RESET)
        return

    if "games" not in data:
        print(RED + "❌ JSON inválido: falta 'games'" + RESET)
        return

    print(f"\nJugador A: {data['player_a']}")
    print(f"Jugador B: {data['player_b']}")
    print(f"Victorias A: {data['player_a_wins']}")
    print(f"Victorias B: {data['player_b_wins']}")
    print(f"Empates:    {data['draws']}")

    games = data["games"]

    # =========================
    # PROCESAR CADA PARTIDA
    # =========================
    for gi, game in enumerate(games, start=1):
        print(f"\n--- PARTIDA {gi} ---")

        # Soportar tu nuevo formato:
        # {
        #   "player_plus1": "...",
        #   "player_minus1": "...",
        #   "history": [...]
        # }
        if isinstance(game, dict) and "history" in game:
            moves = game["history"]
            player_plus1 = game["player_plus1"]
            player_minus1 = game["player_minus1"]
        else:
            print(RED + "Partida vacía o corrupta" + RESET)
            continue

        if not moves:
            print(RED + "Partida sin movimientos" + RESET)
            continue

        last_state, last_col = moves[-1]
        board = [row[:] for row in last_state]

        # Último jugador:
        # movimiento 0 → +1
        # movimiento 1 → -1
        # movimiento 2 → +1 ...
        turn = len(moves) - 1
        last_player = 1 if turn % 2 == 0 else -1

        apply_move(board, last_col, last_player)

        print("\nTablero final con último movimiento aplicado:")
        print_board(board)

        winner = check_winner(board)

        if winner == 1:
            print("Ganador:", BLUE + f"{player_plus1} (+1)" + RESET)
        elif winner == -1:
            print("Ganador:", RED + f"{player_minus1} (-1)" + RESET)
        else:
            print("Resultado: Empate / No 4-en-línea")

# ================================
# PROGRAMA PRINCIPAL
# ================================
directory = "./versus"

if not os.path.exists(directory):
    print(RED + f"❌ La carpeta {directory} no existe" + RESET)
    exit()

# NUEVO PATRÓN: detecta ambos nombres:
pattern = re.compile(r"match_.*_vs_.*\.json$", re.IGNORECASE)

files = [f for f in os.listdir(directory) if pattern.match(f)]

if not files:
    print(RED + "❌ No se encontraron archivos de partidas" + RESET)
    exit()

print(f"🔍 Archivos detectados: {files}")

for fname in files:
    process_file(os.path.join(directory, fname))
