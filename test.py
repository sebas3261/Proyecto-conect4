# test.py
import time
import train  # importa las funciones

t0 = time.time()

# 🔥 Aquí ejecutas realmente los torneos
match_rows, champion_rows = train.run_training(
    runs=10,
    best_of=1,
    fpd=0.5,
    shuffle=True,
    seed=911
)

t1 = time.time()

print(f"\n⏱ Tiempo total: {t1 - t0:.4f} segundos")
