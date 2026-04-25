"""E1.3 — TRUE Bidirectional distillation on Rosbank (CoLES LSTM ↔ LLM LoRA, joint training)."""
import gc
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from distil.data import load_rosbank_dataset
from distil.llm.transaction_serializer import classify_mcc_code
from distil.models import ColesConfig, build_coles_encoder
from distil.reproducibility import seed_everything
from distil.results import save_experiment_result


SEED = 42
DATASET_NAME = "rosbank"
EXPERIMENT_BASE_ID = "E1_3_rosbank"

OUTPUT_DIRECTORY = Path("results/rosbank_true_bidirectional")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
COLES_CHECKPOINT_PATH = Path("results/rosbank_true_latte/coles_baseline.pt")
LLM_CHECKPOINT_PATH = Path("results/rosbank_llm4es/checkpoints/llm4es_lora")

LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.02, max_depth=6, subsample=0.5,
    colsample_bytree=0.75, reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1,
)


def serialize_rosbank_client(transactions, max_transactions: int = 30) -> str:
    """Превращает транзакции клиента в текстовое описание для LLM."""
    if len(transactions) > max_transactions:
        transactions = transactions.tail(max_transactions)
    lines = [f"Client ({len(transactions)} txns):"]
    for _, row in transactions.iterrows():
        direction = "spent" if row["amount"] < 0 else "received"
        category = classify_mcc_code(row["mcc_code"])
        date_string = row["dt"].strftime("%Y-%m-%d") if hasattr(row["dt"], "strftime") else str(row["dt"])
        amount = abs(row["amount"])
        lines.append(f"{date_string}: {direction} {amount:.0f} at {category}")
    return "\n".join(lines)


def extract_coles_embeddings_for_eval(encoder, records, device):
    from ptls.data_load.datasets import inference_data_loader

    encoder.eval()
    inference_loader = inference_data_loader(records, num_workers=0, batch_size=64)
    embedding_chunks = []
    with torch.no_grad():
        for batch in inference_loader:
            embedding_chunks.append(encoder(batch.to(device)).cpu())
    return torch.cat(embedding_chunks).numpy()


def eval_coles_lgbm(encoder, train_records, val_records, test_records,
                    train_targets, val_targets, test_targets, device):
    train_embeddings = extract_coles_embeddings_for_eval(encoder, train_records, device)
    val_embeddings = extract_coles_embeddings_for_eval(encoder, val_records, device)
    test_embeddings = extract_coles_embeddings_for_eval(encoder, test_records, device)

    scaler = MaxAbsScaler()
    scaled_train = scaler.fit_transform(train_embeddings)
    scaled_val = scaler.transform(val_embeddings)
    scaled_test = scaler.transform(test_embeddings)

    from lightgbm import LGBMClassifier
    classifier = LGBMClassifier(**LGBM_PARAMS, random_state=SEED).fit(scaled_train, train_targets)
    val_auc = float(roc_auc_score(val_targets, classifier.predict_proba(scaled_val)[:, 1]))
    test_auc = float(roc_auc_score(test_targets, classifier.predict_proba(scaled_test)[:, 1]))
    return val_auc, test_auc


def get_llm_text_embedding(llm_model, tokenizer, text: str, device):
    """Извлечь LLM embedding одного клиента — mean-pool last 4 hidden layers."""
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False).to(device)
    with torch.set_grad_enabled(llm_model.training):
        model_output = llm_model(**tokenized, output_hidden_states=True)
        hidden_last_4 = torch.stack(model_output.hidden_states[-4:]).mean(0)
        attention_mask = tokenized["attention_mask"][0].unsqueeze(-1).float()
        return (hidden_last_4[0] * attention_mask).sum(0) / attention_mask.sum(0)


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STEP 1: Load Rosbank data")
    print("=" * 60)

    coles_config = ColesConfig.for_dataset(DATASET_NAME)
    full_dataset = load_rosbank_dataset(seed=SEED)

    train_targets_full = full_dataset.train_targets
    test_targets = full_dataset.test_targets

    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset.train_records)),
        test_size=0.1, random_state=SEED, stratify=train_targets_full,
    )
    train_records = [full_dataset.train_records[index] for index in train_indices]
    val_records = [full_dataset.train_records[index] for index in val_indices]
    train_targets = train_targets_full[train_indices]
    val_targets = train_targets_full[val_indices]
    test_records = full_dataset.test_records

    raw_dataframe = pd.read_csv("data/rosbank_train.csv")
    raw_dataframe["dt"] = pd.to_datetime(raw_dataframe["TRDATETIME"], format="%d%b%y:%H:%M:%S")
    raw_dataframe = raw_dataframe.sort_values(["cl_id", "dt"])
    raw_dataframe.rename(columns={"cl_id": "customer_id", "MCC": "mcc_code"}, inplace=True)
    raw_dataframe["mcc_code"] = raw_dataframe["mcc_code"].fillna(0).astype(int)
    raw_dataframe["amount"] = (
        np.sign(raw_dataframe["amount"]) * np.log1p(np.abs(raw_dataframe["amount"]))
    )
    transaction_groups = raw_dataframe.groupby("customer_id")

    train_serialized_texts = [
        serialize_rosbank_client(transaction_groups.get_group(record["customer_id"]))
        for record in train_records
    ]
    print(f"  train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    print("\n" + "=" * 60)
    print("STEP 2: Load CoLES + LLM")
    print("=" * 60)

    sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config).to(device)
    if not COLES_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"CoLES checkpoint not found: {COLES_CHECKPOINT_PATH}\n"
            f"Run: python experiments/rq1_bidirectional/latte/E1_2_rosbank_latte.py first"
        )
    sequence_encoder.load_state_dict(torch.load(COLES_CHECKPOINT_PATH, map_location=device))
    print(f"  CoLES loaded from {COLES_CHECKPOINT_PATH}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", quantization_config=quantization_config, device_map="auto",
    )
    if (LLM_CHECKPOINT_PATH / "adapter_model.safetensors").exists():
        llm_model = PeftModel.from_pretrained(llm_model, str(LLM_CHECKPOINT_PATH))
        print(f"  LLM loaded with LoRA from {LLM_CHECKPOINT_PATH}")
    else:
        lora_configuration = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        llm_model = get_peft_model(llm_model, lora_configuration)
        print("  LLM with fresh LoRA")

    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    llm_hidden_dimension = llm_model.config.hidden_size

    sequence_projection_head = nn.Sequential(
        nn.Linear(coles_config.hidden_size, 256), nn.ReLU(), nn.Linear(256, 128),
    ).to(device)
    text_projection_head = nn.Sequential(
        nn.Linear(llm_hidden_dimension, 256), nn.ReLU(), nn.Linear(256, 128),
    ).to(device)
    sequence_classifier = nn.Sequential(
        nn.Linear(coles_config.hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1),
    ).to(device)
    text_classifier = nn.Sequential(
        nn.Linear(llm_hidden_dimension, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1),
    ).to(device)

    baseline_val_auc, baseline_test_auc = eval_coles_lgbm(
        sequence_encoder, train_records, val_records, test_records,
        train_targets, val_targets, test_targets, device,
    )
    print(f"\n  Baseline CoLES: val={baseline_val_auc:.4f} test={baseline_test_auc:.4f}")

    print("\n" + "=" * 60)
    print("STEP 3: True Bidirectional Fine-tuning")
    print("=" * 60)

    # Best alpha config from Gender E1.3 experiments
    alpha_classification, alpha_contrastive, alpha_mutual = 0.5, 0.3, 0.2
    config_name = f"cls{alpha_classification}_con{alpha_contrastive}_mut{alpha_mutual}"
    print(f"\n--- {config_name} ---")

    for module in [sequence_projection_head, text_projection_head, sequence_classifier, text_classifier]:
        for parameter in module.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    trainable_parameters = (
        list(sequence_encoder.parameters())
        + list(sequence_projection_head.parameters())
        + list(text_projection_head.parameters())
        + list(sequence_classifier.parameters())
        + list(text_classifier.parameters())
    )
    for parameter in llm_model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(trainable_parameters, lr=3e-4, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_auc = baseline_val_auc
    best_test_auc = baseline_test_auc

    from ptls.data_load.datasets import inference_data_loader

    for epoch_index in range(10):
        sequence_encoder.train()
        llm_model.train()
        for module in [sequence_projection_head, text_projection_head, sequence_classifier, text_classifier]:
            module.train()

        shuffled_indices = torch.randperm(len(train_records))
        total_loss = 0.0
        num_batches = 0

        for start_index in range(0, len(train_records), 16):
            batch_indices = shuffled_indices[start_index : start_index + 16].tolist()
            batch_records = [train_records[index] for index in batch_indices]
            batch_texts = [train_serialized_texts[index] for index in batch_indices]
            batch_targets = torch.FloatTensor(
                [record["target"] for record in batch_records]
            ).to(device)

            inference_loader = inference_data_loader(batch_records, num_workers=0, batch_size=32)
            for batch in inference_loader:
                sequence_embedding = sequence_encoder(batch.to(device))

            text_embeddings_list = [
                get_llm_text_embedding(llm_model, tokenizer, text, device)
                for text in batch_texts
            ]
            text_embedding = torch.stack(text_embeddings_list)

            z_sequence = F.normalize(sequence_projection_head(sequence_embedding), dim=1)
            z_text = F.normalize(text_projection_head(text_embedding), dim=1)

            sequence_logits = sequence_classifier(sequence_embedding).squeeze(-1)
            text_logits = text_classifier(text_embedding).squeeze(-1)
            classification_loss_seq = bce_loss(sequence_logits, batch_targets)
            classification_loss_text = bce_loss(text_logits, batch_targets)

            similarity_logits = z_sequence @ z_text.T / 0.07
            diagonal_targets = torch.arange(len(z_sequence), device=device)
            contrastive_loss = (
                F.cross_entropy(similarity_logits, diagonal_targets)
                + F.cross_entropy(similarity_logits.T, diagonal_targets)
            ) / 2

            probability_seq = torch.sigmoid(sequence_logits)
            probability_text = torch.sigmoid(text_logits)
            mutual_distillation_loss = (
                F.binary_cross_entropy(probability_seq, probability_text.detach())
                + F.binary_cross_entropy(probability_text, probability_seq.detach())
            ) / 2

            combined_loss = (
                alpha_classification * (classification_loss_seq + classification_loss_text) / 2
                + alpha_contrastive * contrastive_loss
                + alpha_mutual * mutual_distillation_loss
            )

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            total_loss += float(combined_loss)
            num_batches += 1

        val_auc, test_auc = eval_coles_lgbm(
            sequence_encoder, train_records, val_records, test_records,
            train_targets, val_targets, test_targets, device,
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            torch.save(sequence_encoder.state_dict(), OUTPUT_DIRECTORY / "coles_bidir_best.pt")

        print(
            f"  ep {epoch_index+1}: loss={total_loss/num_batches:.4f}, "
            f"val={val_auc:.4f} test={test_auc:.4f} "
            f"best_val={best_val_auc:.4f} best_test={best_test_auc:.4f}"
        )

    delta_vs_baseline = best_test_auc - baseline_test_auc
    delta_sign = "+" if delta_vs_baseline >= 0 else ""
    print(f"\n  Rosbank: baseline={baseline_test_auc:.4f}, bidir={best_test_auc:.4f} "
          f"({delta_sign}{delta_vs_baseline:.4f})")

    summary_payload = {
        "baseline": baseline_test_auc,
        "bidirectional_best": best_test_auc,
        "baseline_val": baseline_val_auc,
        "best_val": best_val_auc,
    }
    with (OUTPUT_DIRECTORY / "results.json").open("w") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)

    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="LATTE + mutual KL",
        dataset=DATASET_NAME,
        task_type="binary",
        metrics={"roc_auc": best_test_auc, "baseline_roc_auc": baseline_test_auc},
        config={
            "alpha_classification": alpha_classification,
            "alpha_contrastive": alpha_contrastive,
            "alpha_mutual": alpha_mutual,
            "num_epochs": 10,
            "batch_size": 16,
        },
        seed=SEED,
        artifacts={"checkpoint": str(OUTPUT_DIRECTORY / "coles_bidir_best.pt")},
    )

    del llm_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
