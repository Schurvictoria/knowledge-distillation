import subprocess
import sys
from pathlib import Path

from distil.reproducibility import seed_everything

if __name__ == "__main__":
    seed_everything(42)
    here = Path(__file__).parent
    proc = subprocess.run(
        [sys.executable, str(here / "reverse_kl_distill_backend.py"), "--datasets", "age"],
        cwd=here.parents[3],
    )
    sys.exit(proc.returncode)
