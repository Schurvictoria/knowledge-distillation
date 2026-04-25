# E2.2 — Feature-based teacher signal (LLM4ES)

Дистилляция: LLM (Qwen2.5-3B) обучается на serialized транзакциях через QLoRA next-token prediction → извлекаются 2048-d эмбеддинги (mean-pool last 8 hidden layers + masked mean over sequence) → конкатенация с CoLES embeddings → LightGBM downstream.

## Скрипты и canonical конфиги

| Скрипт | Dataset | Train epochs | Размер датасета |
|---|---|---|---|
| `E2_2_gender_llm4es.py` | Gender | 3 | 8.4K clients, 6.85M trans |
| `E2_2_rosbank_llm4es.py` | Rosbank | 3 | 5K clients, 490K trans |
| `E2_2_age_llm4es.py` | Age (v1) | 2 | 30K clients, 27M trans |
| `E2_2_age_llm4es_v2.py` | Age (v2 — **canonical**) | 5 | 30K clients (improved over v1) |

## Почему epoch counts разные

- **Gender / Rosbank: 3 epochs.** Малые датасеты (5–8K клиентов) сходятся за 3 epochs, дальше начинается overfitting.
- **Age v1: 2 epochs.** Больший датасет (30K клиентов × 4-class), но 2 epochs оказалось недостаточно — embeddings были недотренированы.
- **Age v2: 5 epochs.** Корректирует под-обучение v1. Числа в REPORT.md (Age=0.640) — это **v2** (concat с CoLES).

## Reference numbers (REPORT.md, single seed=42)

| Dataset | LLM4ES standalone | CoLES + LLM4ES concat |
|---|---|---|
| Gender (AUC) | 0.679 | **0.864** |
| Rosbank (AUC) | 0.696 | **0.819** |
| Age (Acc) | 0.409 | **0.640** (from v2) |

## Использование v1 vs v2

- v1 оставлен для воспроизведения исторических результатов (если нужно показать "недотренированный")
- **v2 — canonical для статьи.** При запуске E2.2 для Age используй `E2_2_age_llm4es_v2.py`.
