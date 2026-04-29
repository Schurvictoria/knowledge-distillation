import json
import subprocess
from pathlib import Path
from typing import Any

from distil.results.schema import ExperimentResult

_DEFAULT_RESULTS_ROOT = Path("results")

def _detect_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

def _detect_torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return ""

def _detect_ptls_version() -> str:
    try:
        import ptls
        return getattr(ptls, "__version__", "unknown")
    except ImportError:
        return ""

def save_experiment_result(
    *,
    experiment_id: str,
    rq: str,
    method: str,
    dataset: str,
    task_type: str,
    metrics: dict[str, float],
    config: dict[str, Any],
    seed: int = 42,
    runtime_seconds: float = 0.0,
    artifacts: dict[str, str] | None = None,
    notes: str = "",
    output_directory: Path | None = None,
    results_root: Path | None = None,
) -> Path:
    result = ExperimentResult(
        experiment_id=experiment_id,
        rq=rq,
        method=method,
        dataset=dataset,
        task_type=task_type,
        metrics=metrics,
        config=config,
        seed=seed,
        git_commit=_detect_git_commit(),
        torch_version=_detect_torch_version(),
        ptls_version=_detect_ptls_version(),
        runtime_seconds=runtime_seconds,
        artifacts=artifacts or {},
        notes=notes,
    )

    base_directory = results_root or _DEFAULT_RESULTS_ROOT
    target_directory = output_directory or (base_directory / experiment_id)
    target_directory.mkdir(parents=True, exist_ok=True)

    output_file = target_directory / "result.json"
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(result.to_dict(), file_handle, indent=2, ensure_ascii=False)

    return output_file
