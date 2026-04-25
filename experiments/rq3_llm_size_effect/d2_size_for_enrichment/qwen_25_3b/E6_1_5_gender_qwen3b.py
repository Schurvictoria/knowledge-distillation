"""E6.1.5 — Qwen2.5-3B-Instruct kNN-CoT on Gender (D2 size ladder middle)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _d2_runner import run_d2_size_ladder


if __name__ == "__main__":
    run_d2_size_ladder(
        experiment_id="E6_1_5_gender_qwen3b_instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        output_dir_name="gender_d2_qwen25_3b",
    )
