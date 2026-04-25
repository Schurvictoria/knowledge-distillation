from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


_VALID_TASK_TYPES = {"binary", "multiclass", "regression"}
_VALID_RQ_VALUES = {"RQ1", "RQ2-D1", "RQ2-D2", "RQ3"}


@dataclass
class ExperimentResult:
    experiment_id: str
    rq: str
    method: str
    dataset: str
    task_type: str
    metrics: dict[str, float]
    config: dict[str, Any]

    seed: int = 42
    git_commit: str = ""
    torch_version: str = ""
    ptls_version: str = ""
    runtime_seconds: float = 0.0
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {_VALID_TASK_TYPES}, got {self.task_type!r}"
            )
        if self.rq not in _VALID_RQ_VALUES:
            raise ValueError(
                f"rq must be one of {_VALID_RQ_VALUES}, got {self.rq!r}"
            )
        if not self.metrics:
            raise ValueError("metrics dict must contain at least one entry")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
