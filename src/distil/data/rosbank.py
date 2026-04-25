"""
Загрузчик Rosbank датасета (churn prediction).

Что делает: читает data/rosbank_train.csv (одна таблица — и транзакции и labels),
парсит TRDATETIME, кодирует MCC + channel_type + currency + trx_category,
stratified train/test split (90/10, seed=42).
Бинарная задача: предсказать churn (target_flag).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from distil.data._downloads import download_rosbank_data


_DATA_FILE = Path("data") / "rosbank_train.csv"

_MINIMUM_TRANSACTIONS_PER_CLIENT = 15
_TEST_FRACTION = 0.1
_CATEGORICAL_COLUMNS = ["mcc_code", "channel_type", "currency", "trx_category"]
_DATETIME_FORMAT = "%d%b%y:%H:%M:%S"
_NANOSECONDS_PER_DAY = np.timedelta64(1, "D")


@dataclass
class RosbankDataset:
    train_records: list[dict]
    test_records: list[dict]
    feature_dimensions: dict[str, int]
    train_targets: np.ndarray
    test_targets: np.ndarray


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

        datetime_values = client_transactions["dt"].values
        days_since_first = (datetime_values - datetime_values[0]) / _NANOSECONDS_PER_DAY

        record = {
            "customer_id": client_id,
            "target": target_by_client[client_id],
            "event_time": torch.FloatTensor(days_since_first.astype(np.float32)),
            "amount": torch.FloatTensor(client_transactions["amount"].values),
        }
        for column_name, encoder in label_encoders.items():
            encoded_values = encoder.transform(client_transactions[column_name].values) + 1
            record[column_name] = torch.LongTensor(encoded_values)
        built_records.append(record)
    return built_records


def load_rosbank_dataset(seed: int = 42, auto_download: bool = True) -> RosbankDataset:
    if not _DATA_FILE.exists():
        if auto_download:
            download_rosbank_data()
        else:
            raise FileNotFoundError(f"Missing raw data: {_DATA_FILE}.")

    raw_dataframe = pd.read_csv(_DATA_FILE)

    labels_dataframe = raw_dataframe.groupby("cl_id")["target_flag"].max().reset_index()
    labels_dataframe.columns = ["customer_id", "target"]
    target_by_client = dict(zip(labels_dataframe["customer_id"], labels_dataframe["target"]))

    transactions = raw_dataframe.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"}).copy()
    transactions["mcc_code"] = transactions["mcc_code"].fillna(0).astype(int)
    transactions["dt"] = pd.to_datetime(transactions["TRDATETIME"], format=_DATETIME_FORMAT)
    transactions = transactions.sort_values(["customer_id", "dt"])
    transactions["amount"] = np.sign(transactions["amount"]) * np.log1p(np.abs(transactions["amount"]))

    label_encoders = {}
    for column_name in _CATEGORICAL_COLUMNS:
        if column_name not in transactions.columns:
            continue
        transactions[column_name] = transactions[column_name].fillna("UNK").astype(str)
        label_encoders[column_name] = LabelEncoder().fit(transactions[column_name])

    client_ids = labels_dataframe["customer_id"].values
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

    return RosbankDataset(
        train_records=train_records,
        test_records=test_records,
        feature_dimensions=feature_dimensions,
        train_targets=train_targets,
        test_targets=test_targets,
    )
