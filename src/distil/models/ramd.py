"""
Тонкие helpers для RAMD per-dataset скриптов.

RAMD = двухэтапный pipeline:
  1. step1_oof_*.py — OOF предсказания LLM teacher через OpenRouter (kNN-CoT enrichment)
  2. step2_distill.py — CoLES retraining с reverse KL против OOF soft labels

Per-dataset скрипт (E1_4_gender_ramd.py и др.) задаёт `dataset_name` + `teacher_model_key`,
вызывает оба шага через `run_ramd_pipeline()`. Логика самих шагов остаётся в их исходных
скриптах — это сохраняет читаемость и совместимость с CLI запуском напрямую.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


_RAMD_SCRIPTS_DIRECTORY = (
    Path(__file__).resolve().parents[3]
    / "experiments" / "rq1_bidirectional" / "ramd"
)


def _run_step_script(script_name: str, cli_arguments: list[str]) -> None:
    """Запускает один step-скрипт. Падает sys.exit если non-zero."""
    script_path = _RAMD_SCRIPTS_DIRECTORY / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"RAMD step script not found: {script_path}")
    print(f"\n>>> {script_name} {' '.join(cli_arguments)}", flush=True)
    completed_process = subprocess.run(
        [sys.executable, str(script_path), *cli_arguments],
        cwd=_RAMD_SCRIPTS_DIRECTORY.parents[2],
    )
    if completed_process.returncode != 0:
        sys.exit(f"{script_name} failed with code {completed_process.returncode}")


def run_ramd_pipeline(
    dataset_name: str,
    teacher_model_key: str,
    task_type: str,
) -> Optional[float]:
    """Полный RAMD pipeline для одного датасета и одного teacher model.

    Returns: финальная метрика (AUC для binary, accuracy для multiclass) или None
    если step2 не записал ожидаемый JSON.
    """
    if task_type not in {"binary", "multiclass"}:
        raise ValueError(f"task_type must be 'binary' or 'multiclass', got {task_type!r}")

    step1_script = "step1_oof_age.py" if task_type == "multiclass" else "step1_oof_binary.py"
    _run_step_script(step1_script, [
        "--datasets", dataset_name,
        "--models", teacher_model_key,
    ])

    _run_step_script("step2_distill.py", [
        "--datasets", dataset_name,
        "--teacher", teacher_model_key,
    ])

    final_results_path = Path(
        f"results/ramd_kd/{dataset_name}_{teacher_model_key}_results.json"
    )
    if not final_results_path.exists():
        return None

    with final_results_path.open() as file_handle:
        step2_payload = json.load(file_handle)
    return step2_payload.get("best_test", step2_payload.get("ramd_kd_mean"))
