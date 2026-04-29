import subprocess
from pathlib import Path

_DATA_DIRECTORY = Path("data")
_HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/pytorch-lifestream"

def _download_file(url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {target_path.name}...")
    subprocess.run(["curl", "-sL", url, "-o", str(target_path)], check=True)

def _gunzip_to(source_gzipped: Path, target_uncompressed: Path) -> None:
    with target_uncompressed.open("w") as output_handle:
        subprocess.run(["gunzip", "-c", str(source_gzipped)], stdout=output_handle, check=True)
    source_gzipped.unlink(missing_ok=True)

def download_gender_data() -> None:
    base_url = f"{_HUGGINGFACE_BASE_URL}/transactions-gender/resolve/main"
    _DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    transactions_csv = _DATA_DIRECTORY / "transactions.csv"
    transactions_gz = _DATA_DIRECTORY / "transactions.csv.gz"
    if not transactions_csv.exists():
        _download_file(f"{base_url}/transactions.csv.gz?download=true", transactions_gz)
        _gunzip_to(transactions_gz, transactions_csv)

    _download_file(
        f"{base_url}/gender_train.csv?download=true",
        _DATA_DIRECTORY / "gender_train.csv",
    )

def download_rosbank_data() -> None:
    base_url = f"{_HUGGINGFACE_BASE_URL}/rosbank-churn/resolve/main"
    _DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    target_csv = _DATA_DIRECTORY / "rosbank_train.csv"
    if target_csv.exists():
        return

    target_gz = _DATA_DIRECTORY / "rosbank_train.csv.gz"
    _download_file(f"{base_url}/train.csv.gz?download=true", target_gz)
    _gunzip_to(target_gz, target_csv)

def download_age_data() -> None:
    base_url = f"{_HUGGINGFACE_BASE_URL}/age-group-prediction/resolve/main"
    _DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    transactions_csv = _DATA_DIRECTORY / "transactions_train.csv"
    transactions_gz = _DATA_DIRECTORY / "transactions_train.csv.gz"
    if not transactions_csv.exists():
        _download_file(f"{base_url}/transactions_train.csv.gz?download=true", transactions_gz)
        _gunzip_to(transactions_gz, transactions_csv)

    _download_file(
        f"{base_url}/train_target.csv?download=true",
        _DATA_DIRECTORY / "train_target.csv",
    )
