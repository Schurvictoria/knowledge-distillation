"""E5.0 — LATTE on Gender with Qwen2.5-0.5B-Instruct teacher (size ladder, lower bound)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _latte_ladder_runner import run_latte_ladder


if __name__ == "__main__":
    run_latte_ladder(
        experiment_id="E5_0_gender_qwen0_5b_instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        teacher_short="qwen25_0_5b_instruct",
    )
