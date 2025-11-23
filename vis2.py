import os
import json
import re

# ============================================================
# Colores ANSI
# ============================================================
RED = "\033[91m"
BLUE = "\033[94m"
GRAY = "\033[90m"
RESET = "\033[0m"


# ============================================================
# Dibujo del tablero
# ============================================================
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


# ============================================================
# Determinar ganador examinando tablero final
# ============================================================
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
                if all(ok(r + dr*k, c + dc*k) and board[r + dr*k][c + dc*k] == p for k in range(4)):
                    return p

    return 0


# ============================================================
# Aplicar movimiento (solo si hace falta)
# ============================================================
def apply_move(board, col, player):
    if col is None:
        return
    for r in range(5, -1, -1):
        if board[r][col] == 0:
            board[r][col] = player
            return


# ============================================================
# Menú simple
# ============================================================
def menu(options, title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    print("0. Salir / Volver")

    while True:
        try:
            choice = int(input("\nSeleccione: "))
            if 0 <= choice <= len(options):
                return choice
        except:
            pass
        print("❌ Opción inválida")


# ============================================================
# Visualizar jugada por jugada (versión corregida)
# ============================================================
def show_game(game):

    history = game["history"]
    plus1_name = game.get("player_plus1", "Jugador +1")
    minus1_name = game.get("player_minus1", "Jugador -1")

    print("\n" + "="*60)
    print(f"   V I S U A L I Z A N D O    P A R T I D A")
    print("="*60 + "\n")

    n = len(history)
    if n == 0:
        print("Partida vacía")
        return

    for idx in range(n):

        board_before, col = history[idx]

        print(f"\n--- Jugada {idx+1} ---")
        print(f"Columna seleccionada: {col}")

        # DETERMINAR QUIÉN JUGÓ ESTA JUGADA
        if idx < n - 1:
            board_after, _ = history[idx + 1]

            # Detectar cambio exacto entre estados
            player_piece = 0
            for r in range(6):
                if board_before[r][col] != board_after[r][col]:
                    player_piece = board_after[r][col]
                    break

            # fallback raro
            if player_piece == 0:
                flat = sum(board_after, [])
                ones = flat.count(1)
                negs = flat.count(-1)
                player_piece = 1 if ones >= negs else -1

            board_show = board_after

        else:
            # última jugada → aplicar manualmente
            board_show = [row[:] for row in board_before]

            # quién le toca jugar
            flat = sum(board_before, [])
            ones = flat.count(1)
            negs = flat.count(-1)
            player_piece = 1 if ones == negs else -1

            apply_move(board_show, col, player_piece)

        # Mostrar quién jugó
        if player_piece == 1:
            print(f"Jugador que movió: +1 ({plus1_name})")
        else:
            print(f"Jugador que movió: -1 ({minus1_name})")

        # Mostrar tablero
        print("\nTablero después de esta jugada:\n")
        print_board(board_show)

        input("Enter para continuar...")

    # ============================
    # Resultado final
    # ============================
    final_board, _ = history[-1]
    last_flat = sum(final_board, [])
    ones = last_flat.count(1)
    negs = last_flat.count(-1)

    # jugador final "teórico"
    result = check_winner(final_board)

    print("\n======= RESULTADO FINAL =======")
    if result == 1:
        print("Ganador:", BLUE + f"{plus1_name} (+1)" + RESET)
    elif result == -1:
        print("Ganador:", RED + f"{minus1_name} (-1)" + RESET)
    else:
        print("Empate")


# ============================================================
# Procesar archivo completo
# ============================================================
def process_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    games = data["games"]
    options = [f"Partida {i+1}" for i in range(len(games))]

    while True:
        choice = menu(options, f"Archivo: {os.path.basename(path)}")
        if choice == 0:
            break
        show_game(games[choice - 1])


# ============================================================
# MAIN
# ============================================================
def main():
    directory = "./versus"
    pattern = re.compile(r"match_Group [A-H]_vs_Group [A-H]\.json$")
    files = [f for f in os.listdir(directory) if pattern.match(f)]

    if not files:
        print("❌ No se encontraron archivos JSON")
        return

    while True:
        choice = menu(files, "Seleccione un archivo")
        if choice == 0:
            print("Saliendo...")
            return
        process_file(os.path.join(directory, files[choice - 1]))


if __name__ == "__main__":
    main()
