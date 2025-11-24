import json
import os
from multiprocessing import Lock

os.makedirs("training_logs", exist_ok=True)

_log_lock = Lock()

def log_champion(run_id, seed, policy_name):
    entry = {
        "run": run_id,
        "seed": seed,
        "champion": policy_name
    }
    with _log_lock:
        with open("training_logs/champions.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


def log_match(run_id, seed, winner):
    entry = {
        "run": run_id,
        "seed": seed,
        "winner": winner
    }
    with _log_lock:
        with open("training_logs/matches.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
