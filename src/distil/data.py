"""Dataset loading, preprocessing, and feature engineering."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def ensure_gender_data(data_dir: Path = DATA_DIR) -> Path:
    """Download Gender dataset from HuggingFace if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main"
    files = {
        "transactions.csv.gz": "transactions.csv",
        "gender_train.csv": "gender_train.csv",
        "tr_mcc_codes.csv": "tr_mcc_codes.csv",
        "tr_types.csv": "tr_types.csv",
    }
    for remote, local in files.items():
        path = data_dir / local
        if path.exists():
            continue
        gz = remote.endswith(".gz")
        dl_path = data_dir / remote if gz else path
        print(f"Downloading {remote}...")
        subprocess.run(
            ["curl", "-sL", f"{base}/{remote}?download=true", "-o", str(dl_path)],
            check=True,
        )
        if gz:
            subprocess.run(["gunzip", str(dl_path)], check=True)
    return data_dir


def ensure_rosbank_data(data_dir: Path = DATA_DIR) -> Path:
    """Download Rosbank dataset from HuggingFace if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main"
    for name in ["train.csv.gz", "test.csv.gz"]:
        local = name.replace(".gz", "")
        path = data_dir / f"rosbank_{local}"
        if path.exists():
            continue
        dl_path = data_dir / name
        print(f"Downloading rosbank {name}...")
        subprocess.run(
            ["curl", "-sL", f"{base}/{name}?download=true", "-o", str(dl_path)],
            check=True,
        )
        subprocess.run(["gunzip", "-c", str(dl_path)], stdout=open(path, "w"), check=True)
        dl_path.unlink(missing_ok=True)
    return data_dir


def load_gender(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Gender dataset: transactions, labels, MCC codes.

    Returns (transactions, labels, mcc_codes).
    """
    ensure_gender_data(data_dir)
    transactions = pd.read_csv(data_dir / "transactions.csv")
    labels = pd.read_csv(data_dir / "gender_train.csv")
    mcc_codes = pd.read_csv(data_dir / "tr_mcc_codes.csv", sep=";")
    return transactions, labels, mcc_codes


def load_rosbank(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Rosbank dataset: transactions with target.

    Returns (transactions, labels).
    """
    ensure_rosbank_data(data_dir)
    df = pd.read_csv(data_dir / "rosbank_train.csv")
    labels = df.groupby("cl_id")["target_flag"].max().reset_index()
    labels.columns = ["customer_id", "target"]
    df = df.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"})
    return df, labels


def aggregate_features_gender(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into client-level features for Gender dataset."""
    # Only keep clients with known labels
    tx = transactions[transactions["customer_id"].isin(labels["customer_id"])].copy()

    agg = tx.groupby("customer_id").agg(
        n_transactions=("amount", "size"),
        total_spend=("amount", lambda x: x[x < 0].sum()),
        total_income=("amount", lambda x: x[x > 0].sum()),
        mean_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        median_amount=("amount", "median"),
        n_unique_mcc=("mcc_code", "nunique"),
        n_unique_tr_type=("tr_type", "nunique"),
    ).reset_index()

    # MCC distribution features: top-N MCC codes as fraction of transactions
    top_mccs = tx["mcc_code"].value_counts().head(20).index.tolist()
    for mcc in top_mccs:
        col = f"mcc_{mcc}_frac"
        mcc_counts = tx[tx["mcc_code"] == mcc].groupby("customer_id").size()
        total_counts = tx.groupby("customer_id").size()
        agg[col] = agg["customer_id"].map(mcc_counts / total_counts).fillna(0)

    # Transaction type distribution
    top_types = tx["tr_type"].value_counts().head(10).index.tolist()
    for tt in top_types:
        col = f"trtype_{tt}_frac"
        tt_counts = tx[tx["tr_type"] == tt].groupby("customer_id").size()
        total_counts = tx.groupby("customer_id").size()
        agg[col] = agg["customer_id"].map(tt_counts / total_counts).fillna(0)

    agg = agg.merge(labels, on="customer_id", how="inner")
    agg["std_amount"] = agg["std_amount"].fillna(0)
    return agg


def aggregate_features_rosbank(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into client-level features for Rosbank dataset."""
    tx = transactions[transactions["customer_id"].isin(labels["customer_id"])].copy()

    agg = tx.groupby("customer_id").agg(
        n_transactions=("amount", "size"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        median_amount=("amount", "median"),
        n_unique_mcc=("mcc_code", "nunique"),
    ).reset_index()

    top_mccs = tx["mcc_code"].value_counts().head(20).index.tolist()
    for mcc in top_mccs:
        col = f"mcc_{mcc}_frac"
        mcc_counts = tx[tx["mcc_code"] == mcc].groupby("customer_id").size()
        total_counts = tx.groupby("customer_id").size()
        agg[col] = agg["customer_id"].map(mcc_counts / total_counts).fillna(0)

    # Rosbank-specific: trx_category distribution
    if "trx_category" in tx.columns:
        for cat in tx["trx_category"].unique():
            col = f"cat_{cat}_frac"
            cat_counts = tx[tx["trx_category"] == cat].groupby("customer_id").size()
            total_counts = tx.groupby("customer_id").size()
            agg[col] = agg["customer_id"].map(cat_counts / total_counts).fillna(0)

    agg = agg.merge(labels, on="customer_id", how="inner")
    agg["std_amount"] = agg["std_amount"].fillna(0)
    return agg


def prepare_dataset(
    dataset_name: str, data_dir: Path = DATA_DIR, test_size: float = 0.2, seed: int = 42
) -> dict:
    """Load and prepare a dataset, return dict with train/test splits.

    Returns dict with keys:
        X_train, X_test, y_train, y_test: arrays
        feature_names: list of feature column names
        customer_ids_train, customer_ids_test: client IDs
        transactions: raw transaction DataFrame (for text conversion)
        mcc_codes: MCC code descriptions (Gender only)
        labels: full labels DataFrame
    """
    if dataset_name == "gender":
        transactions, labels, mcc_codes = load_gender(data_dir)
        agg = aggregate_features_gender(transactions, labels)
        target_col = "gender"
        extra = {"mcc_codes": mcc_codes}
    elif dataset_name == "rosbank":
        transactions, labels = load_rosbank(data_dir)
        agg = aggregate_features_rosbank(transactions, labels)
        target_col = "target"
        extra = {"mcc_codes": None}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    feature_cols = [c for c in agg.columns if c not in ("customer_id", target_col)]
    X = agg[feature_cols].values.astype(np.float32)
    y = agg[target_col].values.astype(np.int32)
    ids = agg["customer_id"].values

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "customer_ids_train": ids_train,
        "customer_ids_test": ids_test,
        "transactions": transactions,
        "labels": labels if dataset_name == "gender" else labels,
        **extra,
    }
