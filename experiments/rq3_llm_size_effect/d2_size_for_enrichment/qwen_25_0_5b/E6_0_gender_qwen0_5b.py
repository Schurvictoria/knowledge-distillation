"""E6.0 — Qwen2.5-0.5B-Instruct kNN-CoT on Gender (D2 size ladder, lower bound)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _d2_runner import run_d2_size_ladder


if __name__ == "__main__":
    run_d2_size_ladder(
        experiment_id="E6_0_gender_qwen0_5b_instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir_name="gender_d2_qwen25_0_5b",
    )
