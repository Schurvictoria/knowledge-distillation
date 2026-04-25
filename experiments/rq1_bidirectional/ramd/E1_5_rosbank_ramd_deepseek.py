"""E1.5 — RAMD on Rosbank with DeepSeek-V3.2 teacher (kNN-CoT OOF → CoLES distill)."""
from distil.models.ramd import run_ramd_pipeline
from distil.reproducibility import seed_everything
from distil.results import save_experiment_result


SEED = 42
DATASET_NAME = "rosbank"
TEACHER_MODEL_KEY = "deepseek_v3"
EXPERIMENT_BASE_ID = "E1_5_rosbank"


def main() -> None:
    seed_everything(SEED)
    final_auc = run_ramd_pipeline(
        dataset_name=DATASET_NAME,
        teacher_model_key=TEACHER_MODEL_KEY,
        task_type="binary",
    )
    if final_auc is None:
        print("WARNING: step2 did not produce results JSON — skipping save_experiment_result")
        return

    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="RAMD (DeepSeek-V3.2 teacher)",
        dataset=DATASET_NAME,
        task_type="binary",
        metrics={"roc_auc": float(final_auc)},
        config={"teacher_model": TEACHER_MODEL_KEY, "num_folds": 5, "num_neighbors": 10},
        seed=SEED,
    )


if __name__ == "__main__":
    main()
