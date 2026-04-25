"""E5.2-Instruct — LATTE on Gender with Qwen2.5-3B-Instruct teacher (size ladder middle).

Note: separate from canonical E1.2 which uses Qwen2.5-3B (base). This Instruct variant
is part of the unified Qwen2.5-Instruct ladder for fair scaling comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _latte_ladder_runner import run_latte_ladder


if __name__ == "__main__":
    run_latte_ladder(
        experiment_id="E5_2_gender_qwen3b_instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        teacher_short="qwen25_3b_instruct",
    )
