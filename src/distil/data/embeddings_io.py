from pathlib import Path

import numpy as np


_EMBEDDINGS_ROOT = Path("embeddings")


def save_coles_embeddings(
    dataset_name: str,
    seed: int,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    train_customer_ids: np.ndarray,
    test_customer_ids: np.ndarray,
) -> Path:
    output_directory = _EMBEDDINGS_ROOT / dataset_name
    output_directory.mkdir(parents=True, exist_ok=True)

    np.save(output_directory / f"emb_train_seed{seed}.npy", train_embeddings)
    np.save(output_directory / f"emb_test_seed{seed}.npy", test_embeddings)
    np.save(output_directory / f"y_train_seed{seed}.npy", train_targets)
    np.save(output_directory / f"y_test_seed{seed}.npy", test_targets)
    np.save(output_directory / f"cids_train_seed{seed}.npy", train_customer_ids)
    np.save(output_directory / f"cids_test_seed{seed}.npy", test_customer_ids)

    return output_directory


def load_coles_embeddings(
    dataset_name: str,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    embedding_directory = _EMBEDDINGS_ROOT / dataset_name
    return {
        "train_embeddings": np.load(embedding_directory / f"emb_train_seed{seed}.npy"),
        "test_embeddings": np.load(embedding_directory / f"emb_test_seed{seed}.npy"),
        "train_targets": np.load(embedding_directory / f"y_train_seed{seed}.npy"),
        "test_targets": np.load(embedding_directory / f"y_test_seed{seed}.npy"),
        "train_customer_ids": np.load(embedding_directory / f"cids_train_seed{seed}.npy"),
        "test_customer_ids": np.load(embedding_directory / f"cids_test_seed{seed}.npy"),
    }
