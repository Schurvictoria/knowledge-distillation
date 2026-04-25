"""E2.4 — Combined KD on Rosbank (Contrastive + Reverse KL, all three signals together)."""
import subprocess
import sys
from pathlib import Path

from distil.reproducibility import seed_everything


SEED = 42
DATASET_NAME = "rosbank"

_THIS_DIRECTORY = Path(__file__).parent
_REPOSITORY_ROOT = _THIS_DIRECTORY.parents[3]
_BACKEND_SCRIPT = _THIS_DIRECTORY / "combined_kd_backend.py"


def main() -> None:
    seed_everything(SEED)
    completed_process = subprocess.run(
        [sys.executable, str(_BACKEND_SCRIPT), "--datasets", DATASET_NAME],
        cwd=_REPOSITORY_ROOT,
    )
    if completed_process.returncode != 0:
        sys.exit(f"combined_kd_backend failed with code {completed_process.returncode}")


if __name__ == "__main__":
    main()
