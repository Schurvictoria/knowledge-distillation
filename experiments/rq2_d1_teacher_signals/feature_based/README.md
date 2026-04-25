# E2.2 — Feature-based teacher signal (LLM4ES)

Дистилляция: LLM (Qwen2.5-3B) обучается на serialized транзакциях через QLoRA next-token prediction → извлекаются 2048-d эмбеддинги (mean-pool last 8 hidden layers + masked mean over sequence) → конкатенация с CoLES embeddings → LightGBM downstream.

## Скрипты

| Скрипт | Dataset | Train epochs | Размер train |
|---|---|---|---|
| `E2_2_gender_llm4es.py` | Gender | 3 | 8.4K клиентов |
| `E2_2_rosbank_llm4es.py` | Rosbank | 3 | 5K клиентов |
| `E2_2_age_llm4es.py` | Age | 5 | 27K клиентов (full data) |

## Reference numbers (REPORT.md, single seed=42)

| Dataset | LLM4ES standalone | CoLES + LLM4ES concat |
|---|---|---|
| Gender (AUC) | 0.679 | **0.864** |
| Rosbank (AUC) | 0.696 | **0.819** |
| Age (Acc) | 0.413 | **0.640** |

## Output

Все скрипты пишут в `results/<dataset>_llm4es/`:
- `checkpoints/llm4es_lora/` — LoRA adapter weights
- `llm4es_embeddings.npz` — извлечённые эмбеддинги (используется downstream скриптами E1.2 LATTE / E1.3 mutual_KL)
- `*.json` — метрики

## Зависимости

- CoLES baseline должен быть запущен заранее (`run_age_coles.py` и т.д.) — нужны embeddings + cids
- Для Age: `data/transactions_train.csv` + `data/train_target.csv` (~250MB suммарно)
- Для Gender: `data/transactions.csv` + `data/gender_train.csv`
- Для Rosbank: `data/rosbank_train.csv`
