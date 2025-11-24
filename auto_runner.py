import subprocess
import time
import sys
from datetime import datetime

# ----------------------------------------------------------------
# Forzar la codificación UTF-8 en stdout
# ----------------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------------------
# CONFIG — usa SIEMPRE el Python del venv con sys.executable
# ----------------------------------------------------------------
BASE_CMD = [
    sys.executable,      # <-- Garantiza que use el Python del venv
    "train_mp.py",
    "--runs", "10",
    "--games-per-run", "300"
]

START_SEED = 911
WAIT_SECONDS = 3

SUCCESS_MESSAGES = [
    "=== TRAINING FINISHED ===",
    "Logs guardados en logs/training_results.csv",
    "Q-values guardados correctamente."
]

shutdown_requested = False


def run_with_seed(seed: int) -> bool:
    """Ejecuta train_mp.py con la seed especificada y detecta si terminó bien."""
    print(f"\n\n=== Ejecutando run con SEED={seed} ===")

    cmd = BASE_CMD + ["--seed", str(seed)]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    success = False

    for line in process.stdout:
        print(line, end="")
        if any(msg in line for msg in SUCCESS_MESSAGES):
            success = True

    process.wait()
    return success


def main():
    global shutdown_requested

    seed = START_SEED

    print("Presiona CTRL+C para detener después del ciclo actual.\n")

    while True:
        try:
            start_time = datetime.now()
            print(f"\n===== Nuevo ciclo — SEED {seed} — {start_time} =====")

            ok = run_with_seed(seed)

            if ok:
                print(f"[OK] SEED {seed} completado correctamente.")
            else:
                print(f"[ERROR] SEED {seed} terminó con errores o no se detectó finalización correcta.")

            seed += 1

            if shutdown_requested:
                print("\n🟦 Cancelación solicitada — FIN cuando termine el ciclo actual.")
                break

            print(f"Esperando {WAIT_SECONDS} segundos antes del siguiente ciclo…")
            time.sleep(WAIT_SECONDS)

        except KeyboardInterrupt:
            print("\n🟥 CTRL+C detectado — se detendrá al finalizar este ciclo.")
            shutdown_requested = True


if __name__ == "__main__":
    main()
