"""
CoLES baseline — self-supervised contrastive learning of event sequences.

Что делает: TrxEncoder (категории + amount) → RnnSeqEncoder (GRU/LSTM) →
ContrastiveLoss + HardNegativePairSelector через ColesDataset со sample slices.
После обучения: extract_embeddings() даёт hidden_size-мерный вектор на клиента.

Конфиги для трёх датасетов вшиты в ColesDataset.for_dataset() — те же что в
оригинальном dllllb/coles-paper. Менять не надо чтобы числа в REPORT.md совпадали.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np


# Размерности embedding-таблиц для каждого категориального признака
# (взяты из оригинальных coles-paper scenarios)
_GENDER_EMBEDDING_DIMENSIONS = {"mcc_code": 48, "tr_type": 24}
_ROSBANK_EMBEDDING_DIMENSIONS = {"mcc_code": 24, "channel_type": 4, "currency": 4, "trx_category": 4}
_AGE_EMBEDDING_DIMENSIONS = {"small_group": 16}


@dataclass
class ColesConfig:
    hidden_size: int
    rnn_type: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    embedding_dimensions: dict[str, int]
    embeddings_noise: float = 0.003
    split_count: int = 5
    minimum_sequence_length: int = 25
    maximum_sequence_length: int = 200
    lr_step_size: int = 10
    lr_gamma: float = 0.9025

    @classmethod
    def for_dataset(cls, dataset_name: str) -> "ColesConfig":
        if dataset_name == "gender":
            return cls(
                hidden_size=1024,
                rnn_type="gru",
                batch_size=128,
                learning_rate=0.002,
                num_epochs=150,
                embedding_dimensions=dict(_GENDER_EMBEDDING_DIMENSIONS),
                minimum_sequence_length=15,
                maximum_sequence_length=75,
                lr_step_size=10,
            )
        if dataset_name == "rosbank":
            return cls(
                hidden_size=1024,
                rnn_type="lstm",
                batch_size=128,
                learning_rate=0.004,
                num_epochs=60,
                embedding_dimensions=dict(_ROSBANK_EMBEDDING_DIMENSIONS),
                embeddings_noise=0.0003,
                minimum_sequence_length=15,
                maximum_sequence_length=150,
                lr_step_size=10,
            )
        if dataset_name == "age":
            return cls(
                hidden_size=800,
                rnn_type="gru",
                batch_size=64,
                learning_rate=0.001,
                num_epochs=150,
                embedding_dimensions=dict(_AGE_EMBEDDING_DIMENSIONS),
                minimum_sequence_length=25,
                maximum_sequence_length=200,
                lr_step_size=30,
            )
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


def build_coles_encoder(feature_dimensions, config):
    from ptls.nn import RnnSeqEncoder, TrxEncoder

    embeddings_specification = {
        column_name: {"in": feature_dimensions[column_name], "out": config.embedding_dimensions[column_name]}
        for column_name in feature_dimensions
        if column_name in config.embedding_dimensions
    }

    transaction_encoder = TrxEncoder(
        embeddings=embeddings_specification,
        numeric_values={"amount": "identity"},
        embeddings_noise=config.embeddings_noise,
        use_batch_norm_with_lens=True,
    )

    sequence_encoder = RnnSeqEncoder(
        trx_encoder=transaction_encoder,
        hidden_size=config.hidden_size,
        type=config.rnn_type,
        bidir=False,
        trainable_starter="static",
    )

    return sequence_encoder


def train_coles_baseline(sequence_encoder, train_records, config):
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from ptls.data_load.datasets import MemoryMapDataset
    from ptls.frames.coles import CoLESModule, ColesDataset
    from ptls.frames.coles.split_strategy import SampleSlices

    coles_module = CoLESModule(
        seq_encoder=sequence_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=config.learning_rate),
        lr_scheduler_partial=partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        ),
    )

    splitter = SampleSlices(
        split_count=config.split_count,
        cnt_min=config.minimum_sequence_length,
        cnt_max=config.maximum_sequence_length,
    )

    dataset = ColesDataset(MemoryMapDataset(train_records), splitter=splitter)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(coles_module, loader)
    return coles_module, trainer


def extract_embeddings(coles_module, trainer, records, inference_batch_size: int = 64):
    import torch
    from ptls.data_load.datasets import inference_data_loader

    inference_loader = inference_data_loader(records, num_workers=0, batch_size=inference_batch_size)
    prediction_chunks = trainer.predict(coles_module, inference_loader)
    stacked = torch.vstack(prediction_chunks)
    return stacked.cpu().numpy()
