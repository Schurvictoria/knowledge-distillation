# RQ2 Direction 2: Enrichment Type (Structured → LLM)

**Вопрос:** Какой тип знания от structured model лучше помогает LLM в промптинге?

**Фиксированная LLM:** Qwen2.5-7B-Instruct (4-bit local). Размер LLM варьируется в RQ3 (отдельная папка).

## Структура

```
rq2_d2_prompt_enrichment/
├── by_enrichment_type/          # E3.1-E3.5: типы обогащения
│   ├── E3_1_no_enrich.py
│   ├── E3_2_prediction_enrich.py + _age.py
│   └── E3_3_E3_4_E3_5_*_cot_enrichments.py
├── by_strategy/                 # E4.1-E4.3: матрица strategy × enrichment
│   ├── E4_1_E4_2_E4_3_strategy_matrix.py
│   └── E4_3_cot_ablation.py
└── README.md
```

**Где брать canonical Qwen 7B predictions:** базовые числа (E3.1 None, E3.3 SHAP, E4.x None/SHAP columns) приходят из `../rq3_llm_size_effect/E6_2_{gender,rosbank,age}_qwen7b_local.py` — это canonical local Qwen 7B inference. kNN/Combined cells производятся скриптами в `by_enrichment_type/` (CoT-style enrichment через kNN retrieval).

## REPORT.md результаты (Qwen2.5-7B-Instruct, single seed=42)

### Enrichment type (E3.1-E3.5)

| # | Enrichment | Source | Gender | Rosbank | Age |
|---|---|---|---|---|---|
| **E3.1** | None | — | 0.498 | 0.499 | 0.250 |
| **E3.2** | Prediction (XGB confidence) | XGBoost | 0.5083 | 0.5474 | 0.2780 |
| **E3.3** | Explanation (SHAP) | XGBoost | 0.606 | 0.637 | 0.2607 |
| **E3.4** | Retrieval (CoLES kNN) | CoLES embeddings | **0.762** | **0.766** | 0.250 |
| **E3.5** | Combined (SHAP + kNN) | XGBoost + CoLES | 0.745 | 0.751 | 0.2510 |

### Strategy × Enrichment matrix (E4.1-E4.3) — только Gender

| Strategy | None | +SHAP | +kNN | +Both |
|---|---|---|---|---|
| **E4.1** Zero-shot | 0.498 | 0.542 | **0.770** | 0.616 |
| **E4.2** Few-shot | 0.578 | 0.555 | 0.766 | 0.592 |
| **E4.3** CoT | 0.491 | 0.606 | 0.762 | 0.745 |

**Главный finding:** kNN retrieval даёт +26 pp AUC. Это **лучший signal** для structured → LLM transfer.

## Не путать с RQ3 D2

RQ2 D2 (эта папка) = **варьируем enrichment**, фиксированная LLM (Qwen 7B local).

[RQ3 D2](../rq3_llm_size_effect/) = **варьируем модель** (Gemma 4B → DeepSeek 671B), фиксированный enrichment (kNN). Раньше E6.x скрипты лежали здесь в `by_model/`, теперь они в `rq3_llm_size_effect/` для семантической чистоты.
