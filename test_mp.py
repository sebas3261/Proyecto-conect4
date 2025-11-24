# test_mp.py
import time
import train_mp

if __name__ == "__main__":
    print("Iniciando benchmark multicore con 50 torneos...")

    t0 = time.time()

    champions = train_mp.run_training_parallel(
        runs=500,
        shuffle=True,
        seed=911,
        games_per_run= 500
    )

    t1 = time.time()

    print("\n=== TRAINING FINISHED (MULTICORE) ===")
    print("⏱ Tiempo total:", round(t1 - t0, 4), "segundos")
    print("Campeones:", champions)
