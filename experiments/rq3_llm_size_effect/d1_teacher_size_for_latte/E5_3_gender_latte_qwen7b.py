"""E5.3 — LATTE on Gender with Qwen2.5-7B-Instruct teacher (size ladder, top)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _latte_ladder_runner import run_latte_ladder


if __name__ == "__main__":
    run_latte_ladder(
        experiment_id="E5_3_gender_qwen7b_instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        teacher_short="qwen25_7b_instruct",
    )
