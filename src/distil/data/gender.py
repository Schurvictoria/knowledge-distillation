from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


_DATA_DIRECTORY = Path("data")
_TRANSACTION_FILE = _DATA_DIRECTORY / "transactions.csv"
_LABELS_FILE = _DATA_DIRECTORY / "gender_train.csv"

_MINIMUM_TRANSACTIONS_PER_CLIENT = 25
_TEST_FRACTION = 0.1
_CATEGORICAL_COLUMNS = ["mcc_code", "tr_type"]


@dataclass
class GenderDataset:
    train_records: list[dict]
    test_records: list[dict]
    feature_dimensions: dict[str, int]
    train_targets: np.ndarray
    test_targets: np.ndarray


def _parse_transaction_datetime(raw_datetime: str) -> float:
    parts = str(raw_datetime).split(" ", 1)
    day_index = int(parts[0])
    if len(parts) > 1:
        time_components = parts[1].split(":")
        hours = int(time_components[0])
        minutes = int(time_components[1])
        seconds = int(time_components[2])
        seconds_per_day = 86400.0
        fractional_day = (hours * 3600 + minutes * 60 + seconds) / seconds_per_day
        return day_index + fractional_day
    return float(day_index)


def _build_records(
    transactions_grouped_by_client: pd.api.typing.DataFrameGroupBy,
    client_ids: set,
    target_by_client: dict,
    label_encoders: dict[str, LabelEncoder],
) -> list[dict]:
    built_records = []
    for client_id in client_ids:
        if client_id not in target_by_client or client_id not in transactions_grouped_by_client.groups:
            continue
        client_transactions = transactions_grouped_by_client.get_group(client_id)
        if len(client_transactions) < _MINIMUM_TRANSACTIONS_PER_CLIENT:
            continue

        days_since_first = client_transactions["day_float"].values
        days_since_first = (days_since_first - days_since_first[0]).astype(np.float32)

        record = {
            "customer_id": client_id,
            "target": target_by_client[client_id],
            "event_time": torch.FloatTensor(days_since_first),
            "amount": torch.FloatTensor(client_transactions["amount"].values),
        }
        for column_name, encoder in label_encoders.items():
            encoded_values = encoder.transform(client_transactions[column_name].values) + 1
            record[column_name] = torch.LongTensor(encoded_values)
        built_records.append(record)
    return built_records


def load_gender_dataset(seed: int = 42) -> GenderDataset:
    if not _TRANSACTION_FILE.exists() or not _LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Missing raw data. Expected {_TRANSACTION_FILE} and {_LABELS_FILE}.\n"
            "Run: python experiments/rq1_bidirectional/coles/run_gender_coles.py"
        )

    transactions = pd.read_csv(_TRANSACTION_FILE)
    labels = pd.read_csv(_LABELS_FILE)

    transactions = transactions[transactions["customer_id"].isin(labels["customer_id"])].copy()
    transactions["day_float"] = transactions["tr_datetime"].apply(_parse_transaction_datetime)
    transactions = transactions.sort_values(["customer_id", "day_float"])
    transactions["amount"] = np.sign(transactions["amount"]) * np.log1p(np.abs(transactions["amount"]))

    target_by_client = dict(zip(labels["customer_id"], labels["gender"]))

    label_encoders = {}
    for column_name in _CATEGORICAL_COLUMNS:
        transactions[column_name] = transactions[column_name].fillna("UNK").astype(str)
        label_encoders[column_name] = LabelEncoder().fit(transactions[column_name])

    client_ids = labels["customer_id"].values
    targets = np.array([target_by_client[client_id] for client_id in client_ids])
    train_indices, test_indices = train_test_split(
        np.arange(len(client_ids)),
        test_size=_TEST_FRACTION,
        random_state=seed,
        stratify=targets,
    )
    train_client_ids = set(client_ids[train_indices])
    test_client_ids = set(client_ids[test_indices])

    transactions_grouped = transactions.groupby("customer_id")

    train_records = _build_records(transactions_grouped, train_client_ids, target_by_client, label_encoders)
    test_records = _build_records(transactions_grouped, test_client_ids, target_by_client, label_encoders)

    feature_dimensions = {
        column_name: len(encoder.classes_) + 2
        for column_name, encoder in label_encoders.items()
    }

    train_targets = np.array([record["target"] for record in train_records])
    test_targets = np.array([record["target"] for record in test_records])

    return GenderDataset(
        train_records=train_records,
        test_records=test_records,
        feature_dimensions=feature_dimensions,
        train_targets=train_targets,
        test_targets=test_targets,
    )
