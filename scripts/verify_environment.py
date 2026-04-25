"""
Pre-flight check: verify that data, embeddings, and key checkpoints exist
before launching long experiment runs.

Usage:
    python scripts/verify_environment.py
    python scripts/verify_environment.py --strict   # exit 1 on any missing
"""
import argparse
import sys
from pathlib import Path


_DATASETS = ["gender", "rosbank", "age"]

_RAW_DATA_FILES = {
    "gender": ["data/transactions.csv", "data/gender_train.csv"],
    "rosbank": ["data/rosbank_train.csv"],
    "age": ["data/transactions_train.csv", "data/train_target.csv"],
}

_COLES_EMBEDDING_STEMS = ["emb_train", "emb_test", "cids_train", "cids_test", "y_train", "y_test"]


def check_raw_data() -> dict[str, list[Path]]:
    missing = {}
    for dataset_name in _DATASETS:
        absent_paths = [
            Path(relative_path)
            for relative_path in _RAW_DATA_FILES[dataset_name]
            if not Path(relative_path).exists()
        ]
        if absent_paths:
            missing[dataset_name] = absent_paths
    return missing


def check_coles_embeddings(seed: int = 42) -> dict[str, list[Path]]:
    missing = {}
    for dataset_name in _DATASETS:
        embedding_directory = Path(f"embeddings/{dataset_name}")
        absent_paths = [
            embedding_directory / f"{stem}_seed{seed}.npy"
            for stem in _COLES_EMBEDDING_STEMS
            if not (embedding_directory / f"{stem}_seed{seed}.npy").exists()
        ]
        if absent_paths:
            missing[dataset_name] = absent_paths
    return missing


def check_python_dependencies() -> list[str]:
    missing_imports = []
    expected_modules = ["numpy", "pandas", "torch", "sklearn", "lightgbm"]
    optional_modules = ["pytorch_lightning", "ptls", "transformers", "peft"]

    for module_name in expected_modules + optional_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_imports.append(module_name)
    return missing_imports


def check_gpu_available() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {device_name}"
        return False, "CUDA not available — running on CPU will be slow"
    except ImportError:
        return False, "torch not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if anything is missing")
    arguments = parser.parse_args()

    print("=" * 60)
    print("Environment verification")
    print("=" * 60)

    missing_packages = check_python_dependencies()
    if missing_packages:
        print(f"  [WARN] Missing packages: {missing_packages}")
    else:
        print("  [OK]   All Python dependencies importable")

    has_gpu, gpu_message = check_gpu_available()
    print(f"  {'[OK]  ' if has_gpu else '[WARN]'} {gpu_message}")

    print("\n--- Raw data ---")
    missing_raw = check_raw_data()
    for dataset_name in _DATASETS:
        if dataset_name in missing_raw:
            print(f"  [MISS] {dataset_name}: {len(missing_raw[dataset_name])} file(s) absent")
            for missing_path in missing_raw[dataset_name]:
                print(f"         - {missing_path}")
        else:
            print(f"  [OK]   {dataset_name}: raw data present")

    print("\n--- CoLES embeddings (seed=42) ---")
    missing_embeddings = check_coles_embeddings()
    for dataset_name in _DATASETS:
        if dataset_name in missing_embeddings:
            print(f"  [MISS] {dataset_name}: {len(missing_embeddings[dataset_name])} file(s) absent")
        else:
            print(f"  [OK]   {dataset_name}: embeddings present")

    has_any_missing = bool(missing_raw or missing_embeddings or missing_packages)

    print("\n" + "=" * 60)
    if has_any_missing:
        print("Some prerequisites missing. Run CoLES baselines first:")
        for dataset_name in missing_raw or missing_embeddings:
            print(f"  python experiments/rq1_bidirectional/coles/run_{dataset_name}_coles.py")
        if arguments.strict:
            return 1
    else:
        print("All prerequisites satisfied.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
