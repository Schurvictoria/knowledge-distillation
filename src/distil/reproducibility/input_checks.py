"""
Pre-flight проверки что все нужные входные файлы на месте.

Зачем: если запустить LATTE без CoLES baseline, скрипт упадёт с FileNotFoundError
посередине обучения, потеряв время. Эти require_*() функции бросают MissingPrerequisiteError
сразу на старте с подсказкой "Run prerequisite: <script>".

Пример: require_coles_embeddings("gender") проверит что
embeddings/gender/{emb_train,emb_test,cids_train,cids_test,y_train,y_test}_seed42.npy
все существуют.
"""
from pathlib import Path


_RAW_DATA_FILES_BY_DATASET = {
    "gender": ["data/transactions.csv", "data/gender_train.csv"],
    "rosbank": ["data/rosbank_train.csv"],
    "age": ["data/transactions_train.csv", "data/train_target.csv"],
}

_PREREQUISITE_SCRIPT_BY_DATASET = {
    "gender": "experiments/rq1_bidirectional/coles/run_gender_coles.py",
    "rosbank": "experiments/rq1_bidirectional/coles/run_rosbank_coles.py",
    "age": "experiments/rq1_bidirectional/coles/run_age_coles.py",
}


class MissingPrerequisiteError(FileNotFoundError):
    pass


def _format_missing_message(missing_path: Path, prerequisite: str) -> str:
    return (
        f"\n  Missing input: {missing_path}"
        f"\n  Run prerequisite: {prerequisite}\n"
    )


def require_raw_data(dataset: str) -> None:
    if dataset not in _RAW_DATA_FILES_BY_DATASET:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Expected one of: {list(_RAW_DATA_FILES_BY_DATASET)}"
        )

    prerequisite = _PREREQUISITE_SCRIPT_BY_DATASET[dataset]
    for relative_path in _RAW_DATA_FILES_BY_DATASET[dataset]:
        path = Path(relative_path)
        if not path.exists():
            raise MissingPrerequisiteError(_format_missing_message(path, prerequisite))


def require_coles_embeddings(dataset: str, seed: int = 42) -> None:
    if dataset not in _PREREQUISITE_SCRIPT_BY_DATASET:
        raise ValueError(f"Unknown dataset {dataset!r}")

    prerequisite = _PREREQUISITE_SCRIPT_BY_DATASET[dataset]
    embedding_directory = Path(f"embeddings/{dataset}")
    expected_files = ["emb_train", "emb_test", "cids_train", "cids_test", "y_train", "y_test"]

    for file_stem in expected_files:
        path = embedding_directory / f"{file_stem}_seed{seed}.npy"
        if not path.exists():
            raise MissingPrerequisiteError(_format_missing_message(path, prerequisite))


def require_llm4es_embeddings(dataset: str) -> None:
    path = Path(f"results/{dataset}_llm4es/llm4es_embeddings.npz")
    prerequisite = f"experiments/rq2_d1_teacher_signals/feature_based/E2_2_{dataset}_llm4es.py"
    if not path.exists():
        raise MissingPrerequisiteError(_format_missing_message(path, prerequisite))


def require_latte_checkpoint(dataset: str, alpha: float = 0.1) -> None:
    path = Path(f"results/{dataset}_true_latte/coles_finetuned_alpha{alpha}.pt")
    prerequisite = f"experiments/rq1_bidirectional/latte/E1_2_{dataset}_latte.py"
    if not path.exists():
        raise MissingPrerequisiteError(_format_missing_message(path, prerequisite))


def require_ramd_oof(dataset: str, teacher_model: str) -> None:
    path = Path(f"results/ramd_openrouter/{dataset}_{teacher_model}_oof.npz")
    prerequisite = (
        f"experiments/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_*.py "
        f"--models {teacher_model} --datasets {dataset}"
    )
    if not path.exists():
        raise MissingPrerequisiteError(_format_missing_message(path, prerequisite))
