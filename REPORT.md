## RQ1: Bidirectional vs Unidirectional

| Method | Description | Direction | Gender | Rosbank | Age |
|--------|---------------|-----------|--------|---------|-----|
| CoLES baseline | Just a CoLES baseline | No transfer | 0.8626 | 0.8054 | 0.6345 |
| LATTE | CoLES учится у LLM embeddings через contrastive loss. веса LLM не обновляются | LLM → CoLES | 0.8674 | 0.8057 | 0.6429 |
| LATTE + mutual KL | CoLES и LLM учат друг друга одновременно. Оба обновляют веса | LLM ↔ CoLES | 0.8676 | 0.8142 | 0.6363 |
| RAMD (Qwen2.5-7B) | Цикл: CoLES помогает LLM через kNN → LLM помогает CoLES через KL → повторить | LLM ↔ CoLES (loop) | 0.8630 | 0.8074 | — |
| RAMD (DeepSeek-V3.2) | Тот же цикл, сильнее LLM | LLM ↔ CoLES (loop) | **0.8630 ± 0.0006** | **0.8072 ± 0.0034** | OOF running |
| RAMD (GPT-4o) | Тот же цикл, сильнейший LLM | LLM ↔ CoLES (loop) | ? | ? | ? |

## RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

Какой тип учительского сигнала эффективнее при дистилляции?

Student: CoLES GRU/LSTM → LightGBM. Teacher: Qwen2.5-3B (LLM4ES). 5 seeds, mean±std. Train/test split seed=42.

| Signal Type | What student receives from LLM | Method | Gender | Rosbank | Age |
|------------|-------------------------------|--------|--------|---------|-----|
| Response-based | Soft label: "male 73%" | Reverse KL distillation | 0.8633 | 0.8074 | 0.6399 |
| Feature-based | LLM embedding (2048 чисел) как доп. фичи | LLM4ES concat → LGBM | 0.864 | 0.819 | 0.640 |
| Relation-based | "Клиент A и B похожи в LLM space" | Contrastive alignment (LATTE) | 0.8674 | 0.8057 | 0.6429 |
| All three combined | Soft labels + embeddings + contrastive | LATTE + mutual learning + LoRA | 0.8676 | 0.8142 | 0.6363 |

## RQ2 Direction 2: Enrichment Type (Structured → LLM)

Какой тип знания от structured model лучше помогает LLM?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Стратегия: CoT. Один прогон (LLM inference детерминированный). Train/test split seed=42.

| Enrichment | Structured Model | Gender | Rosbank | Age |
|-----------|-----------------|--------|---------|-----|
| None | — | 0.498 | 0.499 | 0.250 |
| Prediction | XGBoost confidence | 0.5083 | 0.5474 | 0.2780 |
| Explanation | XGBoost SHAP | 0.606 | 0.637 | **0.2607** |
| Retrieval | CoLES kNN | 0.762 | 0.766 | 0.250 |
| All combined | XGBoost + CoLES | 0.745 | 0.751 | **0.2510** |

## RQ2 Direction 2: Strategy × Enrichment (матрица)

Зависит ли эффект обогащения от стратегии промптинга?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Dataset: Gender. Один прогон (детерминированный). Train/test split seed=42.

| Strategy | None | + SHAP | + kNN | + Both |
|----------|------|--------|-------|--------|
| Zero-shot | 0.498 | 0.542 | 0.770 | 0.616 |
| Few-shot | 0.578 | 0.555 | 0.766 | 0.592 |
| CoT | 0.491 | 0.606 | 0.762 | 0.745 |


## RQ3: LLM Size Effect


### Direction 1 (LLM → Structured Models, True LATTE distillation)

Влияет ли размер LLM-teacher на качество дистилляции?

Method: True LATTE (contrastive alignment). Student: CoLES → LightGBM. 5 seeds, mean±std. Train/test split seed=42.

| Teacher LLM | Size | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|-------------|------|-------------|---------------|-----------|
| Gemma 3n E2B | 2B | ? | ? | ? |
| Qwen2.5-7B-Instruct | 7B | 0.8674 | 0.8057 | 0.6429 |
| Qwen3.6-35B-A3B | 35B MoE | ? | ? | ? |
| DeepSeek-R1-0528 | 671B MoE | ? | ? | ? |
| GPT-4o | ~200B | ? | ? | ? |

### Direction 2 (Structured Models → LLM, kNN CoT enrichment)

Влияет ли размер LLM на эффективность обогащения промптов?

Method: Zero-shot + kNN CoT (лучшая комбинация из RQ2). Dataset: Gender. Один прогон (детерминированный). Train/test split seed=42.

| LLM | Size | No enrichment | + kNN CoT | Δ |
|-----|------|--------------|-----------|---|
| Gemma 3-4B (proxy for 2B) | 4B | 0.5280 | 0.7669 | +23.9 pp |
| Qwen2.5-7B-Instruct | 7B | 0.498 | 0.762 | +26 pp |
| Qwen3.6-35B-A3B | 35B MoE | 0.5077 | 0.7790 | +27.1 pp |
| DeepSeek-R1-0528 | 671B MoE | 0.5152* | 0.7828* | +26.8 pp* |
| GPT-4o | ~200B | ? | ? | ? |

`*` = без API seed=42. Значения без `*` — строго воспроизводимо (seed=42 в API payload).

### CoT Reasoning Effect

Улучшает ли thinking mode качество LLM при обогащении промптов?

Method: Zero-shot + kNN CoT. Dataset: Gender. Один прогон (детерминированный). Train/test split seed=42.

| Teacher LLM | Size | Thinking=off | Thinking=on | Δ |
|-------------|------|-------------|-------------|---|
| Qwen3.6-35B-A3B | 35B MoE | 0.7138* | 0.7151* | +0.13 pp* |
| DeepSeek-R1-0528 | 671B MoE | N/A | 0.7828* | — |

`*` = запуск БЕЗ API seed=42. DeepSeek thinking=off не поддерживается API (требует reasoning=on).
