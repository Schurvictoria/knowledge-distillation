"""E1.6 — Closed-Loop RAD on Age with DeepSeek-V3.2 teacher.

Iterative RAMD: each round re-annotates with LLM using updated CoLES kNN.
  Iter 0: CoLES_0 embeddings → kNN → LLM OOF → distil → CoLES_1
  Iter 1: CoLES_1 embeddings → kNN* → LLM OOF* → distil → CoLES_2
  Iter 2: CoLES_2 embeddings → kNN** → LLM OOF** → distil → CoLES_3
  Stop if improvement < 5e-4.

Age is 4-class (bins 0-3); uses accuracy as downstream metric.
Requires: OPENROUTER_API_KEY env var.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for run_closed_loop_rad
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root for distil.*

from run_closed_loop_rad import run_closed_loop_rad

from distil.reproducibility import seed_everything
from distil.results import save_experiment_result

SEED = 42
DATASET_NAME = "age"
TEACHER_MODEL_KEY = "deepseek_v3"
EXPERIMENT_BASE_ID = "E1_6_age"


def main() -> None:
    seed_everything(SEED)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set OPENROUTER_API_KEY environment variable.")

    t0 = time.time()
    summary = run_closed_loop_rad(
        dataset_name=DATASET_NAME,
        teacher_model_key=TEACHER_MODEL_KEY,
        api_key=api_key,
        n_iterations=3,
        alpha=0.1,
        n_epochs=15,
        n_seeds=5,
        k=10,
        convergence_thr=5e-4,
    )
    runtime = time.time() - t0

    if summary is None:
        print("WARNING: no results produced — skipping save_experiment_result")
        return

    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="Closed-Loop RAD (DeepSeek-V3.2)",
        dataset=DATASET_NAME,
        task_type="multiclass",
        metrics={"accuracy": summary["best_mean"]},
        config={
            "teacher_model": TEACHER_MODEL_KEY,
            "n_iterations": summary["n_iterations_run"],
            "best_iteration": summary["best_iteration"],
            "num_neighbors": 10,
            "alpha": 0.1,
            "convergence_thr": 5e-4,
        },
        seed=SEED,
        runtime_seconds=runtime,
        notes=(
            f"Closed-Loop RAD: baseline={summary['baseline_mean']:.4f} → "
            f"best={summary['best_mean']:.4f} at iter {summary['best_iteration']}"
        ),
    )


if __name__ == "__main__":
    main()
