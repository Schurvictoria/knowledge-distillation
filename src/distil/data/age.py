from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from distil.data._downloads import download_age_data

_DATA_DIRECTORY = Path("data")
_TRANSACTION_FILE = _DATA_DIRECTORY / "transactions_train.csv"
_LABELS_FILE = _DATA_DIRECTORY / "train_target.csv"

_MINIMUM_TRANSACTIONS_PER_CLIENT = 25
_TEST_FRACTION = 0.1
_CATEGORICAL_COLUMNS = ["small_group"]

@dataclass
class AgeDataset:
    train_records: list[dict]
    test_records: list[dict]
    feature_dimensions: dict[str, int]
    train_targets: np.ndarray
    test_targets: np.ndarray

def _build_records(
    transactions_grouped_by_client: pd.api.typing.DataFrameGroupBy,
    client_ids: set,
    target_by_client: dict,
    label_encoder: LabelEncoder,
) -> list[dict]:
    built_records = []
    for client_id in client_ids:
        if client_id not in target_by_client or client_id not in transactions_grouped_by_client.groups:
            continue
        client_transactions = transactions_grouped_by_client.get_group(client_id)
        if len(client_transactions) < _MINIMUM_TRANSACTIONS_PER_CLIENT:
            continue

        days_since_first = client_transactions["trans_date"].values.astype(np.float32)
        days_since_first = days_since_first - days_since_first[0]

        encoded_small_groups = label_encoder.transform(client_transactions["small_group"].values) + 1

        record = {
            "customer_id": client_id,
            "target": target_by_client[client_id],
            "event_time": torch.FloatTensor(days_since_first),
            "amount": torch.FloatTensor(client_transactions["amount_rur"].values),
            "small_group": torch.LongTensor(encoded_small_groups),
        }
        built_records.append(record)
    return built_records

def load_age_dataset(seed: int = 42, auto_download: bool = True) -> AgeDataset:
    if not _TRANSACTION_FILE.exists() or not _LABELS_FILE.exists():
        if auto_download:
            download_age_data()
        else:
            raise FileNotFoundError(
                f"Missing raw data. Expected {_TRANSACTION_FILE} and {_LABELS_FILE}."
            )

    transactions = pd.read_csv(_TRANSACTION_FILE)
    labels = pd.read_csv(_LABELS_FILE)

    target_by_client = dict(zip(labels["client_id"], labels["bins"]))

    transactions = transactions.sort_values(["client_id", "trans_date"])
    transactions["amount_rur"] = (
        np.sign(transactions["amount_rur"]) * np.log1p(np.abs(transactions["amount_rur"]))
    )
    transactions["small_group"] = transactions["small_group"].fillna(0).astype(str)

    small_group_encoder = LabelEncoder().fit(transactions["small_group"])

    client_ids = labels["client_id"].values
    targets = np.array([target_by_client[client_id] for client_id in client_ids])
    train_indices, test_indices = train_test_split(
        np.arange(len(client_ids)),
        test_size=_TEST_FRACTION,
        random_state=seed,
        stratify=targets,
    )
    train_client_ids = set(client_ids[train_indices])
    test_client_ids = set(client_ids[test_indices])

    transactions_grouped = transactions.groupby("client_id")

    train_records = _build_records(transactions_grouped, train_client_ids, target_by_client, small_group_encoder)
    test_records = _build_records(transactions_grouped, test_client_ids, target_by_client, small_group_encoder)

    feature_dimensions = {"small_group": len(small_group_encoder.classes_) + 2}

    train_targets = np.array([record["target"] for record in train_records])
    test_targets = np.array([record["target"] for record in test_records])

    return AgeDataset(
        train_records=train_records,
        test_records=test_records,
        feature_dimensions=feature_dimensions,
        train_targets=train_targets,
        test_targets=test_targets,
    )
