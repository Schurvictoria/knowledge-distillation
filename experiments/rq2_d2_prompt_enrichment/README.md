# RQ2 Direction 2: Enrichment Type (Structured → LLM)

**Вопрос:** Какой тип знания от structured model лучше помогает LLM в промптинге?

## Подпапки

| Папка | Содержимое |
|---|---|
| [`by_model/`](by_model/) | Runners под конкретные LLM (DeepSeek, Qwen3.6, GLM-4.7, Qwen2.5-7B) |
| [`by_enrichment_type/`](by_enrichment_type/) | Скрипты с разными типами обогащения (SHAP/kNN/Both/None) |
| [`by_strategy/`](by_strategy/) | Ablation по стратегиям промптинга (zero/few/CoT × enrichment) |

## Main результаты (Qwen2.5-7B-Instruct 4-bit, CoT strategy)

| Enrichment | Gender | Rosbank | Age |
|---|---|---|---|
| None | 0.498 | 0.499 | 0.249 |
| Prediction (XGB conf) | ? | ? | ? |
| Explanation (SHAP) | 0.606 | 0.637 | ? |
| Retrieval (kNN) | 0.762 | 0.766 | 0.250 |
| Combined | 0.745 | 0.751 | ? |

## Strategy × Enrichment матрица (Gender, Qwen2.5-7B)

| Strategy | None | +SHAP | +kNN | +Both |
|---|---|---|---|---|
| Zero-shot | 0.498 | 0.542 | 0.770 | 0.616 |
| Few-shot | 0.578 | 0.555 | 0.766 | 0.592 |
| CoT | 0.491 | 0.606 | 0.762 | 0.745 |

Скрипт: `by_strategy/run_strategy_matrix.py`

## TODO
- Prediction-only enrichment (3 cells): модифицировать `by_enrichment_type/run_gender_rosbank_cot.py` (убрать SHAP факторы, оставить только prediction)
- Age SHAP + Combined (2 cells): проверить `by_enrichment_type/run_age_structured_cot.py`
