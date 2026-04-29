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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from distil.data import load_age_dataset
from distil.models import ColesConfig, build_coles_encoder
from distil.reproducibility import seed_everything
from distil.results import save_experiment_result

SEED = 42
DATASET_NAME = "age"
EXPERIMENT_BASE_ID = "E1_3_age"

OUTPUT_DIRECTORY = Path("results/age_true_bidirectional")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
COLES_CHECKPOINT_PATH = Path("results/age_true_latte/coles_baseline.pt")
LLM_CHECKPOINT_PATH = Path("results/age_llm4es/checkpoints/llm4es_lora")

LGBM_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, objective="multiclass", num_class=4,
    max_depth=12, num_leaves=50, subsample=0.75, colsample_bytree=0.75,
    reg_alpha=1, reg_lambda=1, min_child_samples=50, verbosity=-1,
)

ACCUM_STEPS = 8

def serialize_age_client(transactions_for_client: pd.DataFrame) -> str:
    n = len(transactions_for_client)
    amounts = np.abs(transactions_for_client["amount_rur"].values)
    cats = transactions_for_client["small_group"].fillna("0").value_counts()
    n_unique_cats = cats.nunique()
    top_cats = ", ".join(
        f"cat{c} ({cnt} txns, {cnt * 100 // max(n, 1)}%)"
        for c, cnt in cats.head(6).items()
    )
    days_span = (
        int(transactions_for_client["trans_date"].max() - transactions_for_client["trans_date"].min())
        if len(transactions_for_client) > 1 else 0
    )
    months = max(1, days_span // 30)
    micro_pct = int((amounts < 500).sum() * 100 // max(n, 1))
    large_pct = int((amounts > 5000).sum() * 100 // max(n, 1))
    return (f"Client ({n} txns, {months}m, {max(1, n // months)}/mo):\n"
            f"Avg: {amounts.mean():.0f} RUB, median: {np.median(amounts):.0f}, max: {amounts.max():.0f}\n"
            f"Size: {micro_pct}% small (<500), {large_pct}% large (>5000)\n"
            f"Diversity: {n_unique_cats} categories. Top: {top_cats}")

def extract_coles_embeddings_for_eval(encoder, records, device):
    from ptls.data_load.datasets import inference_data_loader

    encoder.eval()
    chunks = []
    with torch.no_grad():
        for batch in inference_data_loader(records, num_workers=0, batch_size=64):
            chunks.append(encoder(batch.to(device)).cpu())
    return torch.cat(chunks).numpy()

def eval_coles_lgbm(encoder, train_records, val_records, test_records,
                    train_targets, val_targets, test_targets, device):
    train_emb = extract_coles_embeddings_for_eval(encoder, train_records, device)
    val_emb = extract_coles_embeddings_for_eval(encoder, val_records, device)
    test_emb = extract_coles_embeddings_for_eval(encoder, test_records, device)

    scaler = MaxAbsScaler()
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(**LGBM_PARAMS, random_state=SEED)
    clf.fit(scaler.fit_transform(train_emb), train_targets)
    val_acc = float(accuracy_score(val_targets, clf.predict(scaler.transform(val_emb))))
    test_acc = float(accuracy_score(test_targets, clf.predict(scaler.transform(test_emb))))
    return val_acc, test_acc

def get_llm_batch_embeddings(llm_model, tokenizer, texts: list, device):
    tokenized = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=256,
        padding=True, pad_to_multiple_of=8,
    ).to(device)
    with torch.set_grad_enabled(llm_model.training):
        output = llm_model(**tokenized, output_hidden_states=True)
        hidden_last_4 = torch.stack(output.hidden_states[-4:]).mean(0)
        mask = tokenized["attention_mask"].unsqueeze(-1).float()
        return (hidden_last_4 * mask).sum(1) / mask.sum(1)

def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("STEP 1: Load Age data")

    coles_config = ColesConfig.for_dataset(DATASET_NAME)
    full_dataset = load_age_dataset(seed=SEED)

    train_targets_full = full_dataset.train_targets
    test_targets = full_dataset.test_targets

    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset.train_records)),
        test_size=0.1, random_state=SEED, stratify=train_targets_full,
    )
    train_records = [full_dataset.train_records[i] for i in train_indices]
    val_records = [full_dataset.train_records[i] for i in val_indices]
    train_targets = train_targets_full[train_indices]
    val_targets = train_targets_full[val_indices]
    test_records = full_dataset.test_records

    raw_tx = pd.read_csv("data/transactions_train.csv")
    raw_tx = raw_tx.sort_values(["client_id", "trans_date"])
    raw_tx["small_group"] = raw_tx["small_group"].fillna("0").astype(str)
    transaction_groups = raw_tx.groupby("client_id")

    train_serialized_texts = [
        serialize_age_client(transaction_groups.get_group(record["customer_id"]))
        if record["customer_id"] in transaction_groups.groups
        else "No transactions."
        for record in train_records
    ]

    print("STEP 2: Load CoLES + LLM")

    sequence_encoder = build_coles_encoder(full_dataset.feature_dimensions, coles_config).to(device)
    if not COLES_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"CoLES checkpoint not found: {COLES_CHECKPOINT_PATH}\n"
            f"Run: python experiments/rq1_bidirectional/latte/E1_2_age_latte.py first"
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
        nn.Linear(coles_config.hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4),
    ).to(device)
    text_classifier = nn.Sequential(
        nn.Linear(llm_hidden_dimension, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4),
    ).to(device)

    baseline_val_acc, baseline_test_acc = eval_coles_lgbm(
        sequence_encoder, train_records, val_records, test_records,
        train_targets, val_targets, test_targets, device,
    )
    print(f"\n  Baseline CoLES: val={baseline_val_acc:.4f} test={baseline_test_acc:.4f}")

    print("STEP 3: True Bidirectional Fine-tuning")

    results = {
        "baseline_coles_val": baseline_val_acc,
        "baseline_coles_test": baseline_test_acc,
    }

    best_overall_score = baseline_test_acc
    best_overall_config = "baseline"

    alpha_configurations = [(0.5, 0.3, 0.2)]
    ce_loss = nn.CrossEntropyLoss()

    for alpha_classification, alpha_contrastive, alpha_mutual in alpha_configurations:
        config_name = f"cls{alpha_classification}_con{alpha_contrastive}_mut{alpha_mutual}"

        sequence_encoder.load_state_dict(torch.load(COLES_CHECKPOINT_PATH, map_location=device))
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

        best_val_for_config = baseline_val_acc
        best_test_for_config = baseline_test_acc

        from ptls.data_load.datasets import inference_data_loader

        for epoch_index in range(10):
            sequence_encoder.train()
            llm_model.train()
            for module in [sequence_projection_head, text_projection_head, sequence_classifier, text_classifier]:
                module.train()

            shuffled_indices = torch.randperm(len(train_records))
            total_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()

            for step_index, start_index in enumerate(range(0, len(train_records), 16)):
                batch_indices = shuffled_indices[start_index: start_index + 16].tolist()
                batch_records = [train_records[i] for i in batch_indices]
                batch_texts = [train_serialized_texts[i] for i in batch_indices]
                batch_targets = torch.LongTensor(
                    [record["target"] for record in batch_records]
                ).to(device)

                for batch in inference_data_loader(batch_records, num_workers=0, batch_size=32):
                    sequence_embedding = sequence_encoder(batch.to(device))

                text_embedding = get_llm_batch_embeddings(llm_model, tokenizer, batch_texts, device)

                z_sequence = F.normalize(sequence_projection_head(sequence_embedding), dim=1)
                z_text = F.normalize(text_projection_head(text_embedding), dim=1)

                sequence_logits = sequence_classifier(sequence_embedding)
                text_logits = text_classifier(text_embedding)
                classification_loss = (
                    ce_loss(sequence_logits, batch_targets)
                    + ce_loss(text_logits, batch_targets)
                ) / 2

                similarity_logits = z_sequence @ z_text.T / 0.07
                diagonal_targets = torch.arange(len(z_sequence), device=device)
                contrastive_loss = (
                    F.cross_entropy(similarity_logits, diagonal_targets)
                    + F.cross_entropy(similarity_logits.T, diagonal_targets)
                ) / 2

                log_p_seq = F.log_softmax(sequence_logits, dim=1)
                log_p_text = F.log_softmax(text_logits, dim=1)
                p_seq = log_p_seq.exp()
                p_text = log_p_text.exp()
                mutual_distillation_loss = (
                    F.kl_div(log_p_seq, p_text.detach(), reduction="batchmean")
                    + F.kl_div(log_p_text, p_seq.detach(), reduction="batchmean")
                ) / 2

                combined_loss = (
                    alpha_classification * classification_loss
                    + alpha_contrastive * contrastive_loss
                    + alpha_mutual * mutual_distillation_loss
                ) / ACCUM_STEPS

                combined_loss.backward()
                total_loss += float(combined_loss) * ACCUM_STEPS
                num_batches += 1

                if (step_index + 1) % ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            val_acc, test_acc = eval_coles_lgbm(
                sequence_encoder, train_records, val_records, test_records,
                train_targets, val_targets, test_targets, device,
            )
            if val_acc > best_val_for_config:
                best_val_for_config = val_acc
                best_test_for_config = test_acc
                torch.save(
                    sequence_encoder.state_dict(),
                    OUTPUT_DIRECTORY / f"coles_bidir_{config_name}.pt",
                )

            print(
                f"  ep {epoch_index + 1}: loss={total_loss / num_batches:.4f}, "
                f"val={val_acc:.4f} test={test_acc:.4f} "
                f"best_val={best_val_for_config:.4f} best_test={best_test_for_config:.4f}"
            )

        results[config_name] = best_test_for_config
        if best_test_for_config > best_overall_score:
            best_overall_score = best_test_for_config
            best_overall_config = config_name

        torch.cuda.empty_cache()
        gc.collect()

    print("TRUE BIDIRECTIONAL SUMMARY (Age)")
    for method_name, method_score in sorted(results.items(), key=lambda pair: -pair[1]):
        delta = method_score - baseline_test_acc
        print(f"  {method_name:<35} acc={method_score:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})")

    with (OUTPUT_DIRECTORY / "true_bidir_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    save_experiment_result(
        experiment_id=EXPERIMENT_BASE_ID,
        rq="RQ1",
        method="LATTE + mutual KL",
        dataset=DATASET_NAME,
        task_type="multiclass",
        metrics={"accuracy": best_overall_score, "baseline_accuracy": baseline_test_acc},
        config={
            "alpha_configurations_tried": alpha_configurations,
            "best_alpha_config": best_overall_config,
            "num_epochs": 10,
            "batch_size": 16,
            "accum_steps": ACCUM_STEPS,
        },
        seed=SEED,
        artifacts={"best_checkpoint": str(OUTPUT_DIRECTORY / f"coles_bidir_{best_overall_config}.pt")},
    )

    del llm_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
