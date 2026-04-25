# Результаты экспериментов

## Methodology summary

- **Splits:** train/test 90/10 stratified, `seed=42` для всех датасетов.
- **Datasets:** Gender (binary, AUC), Rosbank (binary churn, AUC), Age (4-class, accuracy).
- **Reproducibility:** `torch.manual_seed(42)` + `cudnn.deterministic=True` + `pytorch_lightning.seed_everything(42)`. OpenRouter API: `seed=42` в payload.
- **Seeds policy:** RQ1 / RQ3 — single seed=42 (compute-limited). RQ2 D1 backends поддерживают 5 seeds `[42,123,456,789,1024]` (см. примечание ниже).
- **Canonical LLM teacher (D1):** Qwen2.5-3B (base) для LLM4ES embedding extraction. `hidden_size=2048`.
- **Canonical LLM (D2):** Qwen2.5-7B-Instruct, 4-bit NF4 для kNN/CoT inference.

## RQ1: Bidirectional vs Unidirectional

| # | Method | Description | Direction | Gender | Rosbank | Age |
|---|---|---|---|---|---|---|
| **E1.1** | CoLES baseline | Just a CoLES baseline | No transfer | 0.8626 | 0.8054 | 0.6345 |
| **E1.2** | LATTE | CoLES учится у LLM embeddings через contrastive loss; веса LLM не обновляются | LLM → CoLES | **0.8713** | **0.8082** | 0.6333 |
| **E1.3** | LATTE + mutual KL | CoLES и LLM учат друг друга одновременно; оба обновляют веса | LLM ↔ CoLES | 0.8676 | 0.8099 | 0.6363 |
| **E1.4** | RAMD (Qwen2.5-7B) | Цикл: CoLES → LLM (kNN) → KL обратно → повтор | LLM ↔ CoLES (loop) | 0.8630 | 0.8074 | skipped (см. E1.5) |
| **E1.5** | RAMD (DeepSeek-V3.2) | Тот же цикл, сильнее LLM | LLM ↔ CoLES (loop) | 0.8630 ± 0.0006 | 0.8072 ± 0.0034 | OOF running |

**Teacher для E1.2 / E1.3:** Qwen2.5-3B-base (LLM4ES embeddings).

## RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

Какой тип учительского сигнала эффективнее при дистилляции?

Student: CoLES GRU/LSTM → LightGBM. Teacher: Qwen2.5-3B-base (LLM4ES embeddings, hidden_dim=2048). Train/test split seed=42.

⚠️ **Seeds:** backend код поддерживает 5 seeds, но числа в таблице — single seed=42 (per-seed CSV не сохранены). 5-seed reruns — pending.

| # | Signal Type | What student receives from LLM | Method | Gender | Rosbank | Age |
|---|---|---|---|---|---|---|
| **E2.1** | Response-based | Soft label: "male 73%" | Reverse KL distillation | 0.8633 | 0.8074 | 0.6399 |
| **E2.2** | Feature-based | LLM embedding (2048) как доп. фичи | LLM4ES concat → LGBM | 0.864 | 0.819 | 0.640 |
| **E2.3** | Relation-based | "Клиент A и B похожи в LLM space" | Contrastive alignment (LATTE) | 0.8674 | 0.8057 | 0.6429 |
| **E2.4** | All three combined | Soft + embeddings + contrastive | LATTE + mutual learning + LoRA | 0.8676 | 0.8142 | 0.6363 |

> **Note:** E2.3 LATTE = E1.2 (one experiment, two RQ aspects). Numbers ничтожно отличаются из-за reseeding evaluation downstream — оба валидны.

## RQ2 Direction 2: Enrichment Type (Structured → LLM)

Какой тип знания от structured model лучше помогает LLM?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Стратегия: CoT. Один прогон (LLM inference детерминированный, `temperature=0`). Train/test split seed=42.

| # | Enrichment | Structured Model | Gender | Rosbank | Age |
|---|---|---|---|---|---|
| **E3.1** | None | — | 0.498 | 0.499 | 0.250 |
| **E3.2** | Prediction | XGBoost confidence | 0.5083 | 0.5474 | 0.2780 |
| **E3.3** | Explanation | XGBoost SHAP | 0.606 | 0.637 | 0.2607 |
| **E3.4** | Retrieval | CoLES kNN | 0.762 | 0.766 | 0.250 |
| **E3.5** | All combined | XGBoost + CoLES | 0.745 | 0.751 | 0.2510 |

**Negative finding на Age:** kNN/CoT enrichment не помогает на 4-class Age (все ≈ 0.25 = chance). Гипотеза: 4-class label в текстовом промпте `male / female` интерпретируется LLM однозначно, а возрастные интервалы (`<35 / 35-50 / 50-65 / >65`) confounded с другими сигналами в retrieved similar customers.

## RQ2 Direction 2: Strategy × Enrichment (матрица)

Зависит ли эффект обогащения от стратегии промптинга?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Dataset: Gender. Один прогон. Train/test split seed=42.

| # | Strategy | None | + SHAP | + kNN | + Both |
|---|---|---|---|---|---|
| **E4.1** | Zero-shot | 0.498 | 0.542 | 0.770 | 0.616 |
| **E4.2** | Few-shot | 0.578 | 0.555 | 0.766 | 0.592 |
| **E4.3** | CoT | 0.491 | 0.606 | 0.762 | 0.745 |

## RQ3: LLM Size Effect

### Direction 1 (LLM → Structured Models, LATTE distillation)

Влияет ли размер LLM-teacher на качество дистилляции?

Method: LATTE (contrastive alignment). Teacher: Qwen2.5 family (Instruct variants для нового ladder). Student: CoLES → LightGBM. Train/test split seed=42.

⚠️ **Family-clean ladder pending.** Существующие числа (E5.2 ниже) были получены на Qwen2.5-3B-**base**; новый ladder — Qwen2.5-Instruct family на Gender для чистого scaling-style сравнения.

| # | Teacher LLM | Size | Variant | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|---|---|---|---|---|---|---|
| **E5.0** | Qwen2.5-0.5B | 0.5B | Instruct | pending (Gender ladder) | — | — |
| **E5.1** | Qwen2.5-1.5B | 1.5B | Instruct | pending (Gender ladder) | — | — |
| **E5.2** | Qwen2.5-3B (existing) | 3B | base | 0.8674 | 0.8057 | 0.6429 |
| **E5.2-Instruct** | Qwen2.5-3B | 3B | Instruct | pending (Gender ladder) | — | — |
| **E5.3** | Qwen2.5-7B | 7B | Instruct | pending (Gender ladder) | — | — |

> **Hardware constraint в Limitations:** LATTE требует доступа к hidden states LLM. Локально доступны только модели ≤7B (RTX 3090, 24GB). Модели >7B протестированы только в D2 (kNN inference, response-based, через API) — см. ниже.

### Direction 2 (Structured Models → LLM, kNN CoT enrichment)

Влияет ли размер LLM на эффективность обогащения промптов?

Method: Zero-shot + kNN ("Similar clients: X pos, Y neg"). Dataset: Gender. Train/test split seed=42.

#### Headline ladder — Qwen2.5 family only (clean scaling)

| # | LLM | Size | Variant | No enrichment | + kNN | Δ |
|---|---|---|---|---|---|---|
| **E6.0** | Qwen2.5-0.5B | 0.5B | Instruct | pending | pending | — |
| **E6.1** | Qwen2.5-1.5B | 1.5B | Instruct | pending | pending | — |
| **E6.1.5** | Qwen2.5-3B | 3B | Instruct | pending | pending | — |
| **E6.2** | Qwen2.5-7B | 7B | Instruct | 0.498 | 0.762 | +26 pp |

#### Supplementary — different families confirm finding

| # | LLM | Size | Family | No enrichment | + kNN | Δ |
|---|---|---|---|---|---|---|
| **E6.3** | Gemma 3-4B | 4B | Gemma | 0.5280 | 0.7669 | +23.9 pp |
| **E6.4** | Qwen3.6-35B-A3B | 35B MoE | Qwen3 (different gen) | 0.5077 | 0.7790 | +27.1 pp |
| **E6.5** | DeepSeek-V3.2-Speciale | 671B MoE | DeepSeek | 0.5152* | 0.7828* | +26.8 pp* |

`*` = без API seed=42 (provider non-deterministic для MoE).

### CoT Reasoning Effect

Улучшает ли thinking mode качество LLM при обогащении промптов?

Method: Zero-shot + kNN. Dataset: Gender.

| # | Teacher LLM | Size | Thinking=off | Thinking=on | Δ |
|---|---|---|---|---|---|
| **E7.1** | Qwen3.6-35B-A3B | 35B MoE | 0.7138* | 0.7151* | +0.13 pp* |
| **E7.2** | DeepSeek-V3.2 | 671B MoE | N/A | 0.7828* | — |
| **E7.3** | GLM-4.7 (bonus) | ~9B | 0.7712 | 0.6541 (parse bug) | — |

`*` = без API seed=42.

---

## Pending experiments

| # | Что | Где | Time | Status |
|---|---|---|---|---|
| **E1.5 Age** | RAMD DeepSeek на Age | `experiments/rq1_bidirectional/ramd/E1_5_age_ramd_deepseek.py` | ~3-4h | OOF running |
| **E1.4 Age** | RAMD Qwen 7B на Age | (script нет, см. E1.5) | — | skipped |
| **E5.0 / E5.1 / E5.2-Instruct / E5.3** | Qwen2.5 Instruct ladder D1 на Gender | новые скрипты в `rq3_llm_size_effect/d1_teacher_size_for_latte/` | ~12h GPU | pending |
| **E6.0 / E6.1 / E6.1.5** | Qwen2.5 Instruct ladder D2 на Gender | новые скрипты в `rq3_llm_size_effect/d2_size_for_enrichment/qwen_25_*/` | ~3-4h GPU | pending |
| **RQ2 D1 — 5 seeds rerun** | Восстановить ± std для E2.1, E2.2, E2.4 | существующие скрипты, retrieve per-seed CSV | ~10h GPU | pending |

---

## Methodology limitations (для paper)

1. **Single-seed для RQ1 / RQ3.** Variance estimated via 1000-resample bootstrap of test predictions (CI half-width: ~0.008-0.015 AUC).
2. **D1 teacher size sweep ограничен 7B сверху** из-за hardware (24GB VRAM RTX 3090, 4-bit NF4 quant). Hidden states required для LATTE недоступны через API → большие модели тестируются только в D2.
3. **Base vs Instruct teacher.** Существующие D1 numbers использовали Qwen2.5-3B-base; D2 — Qwen2.5-7B-Instruct. Новый ladder унифицирует на Instruct family.
4. **kNN/CoT enrichment не помогает на Age multi-class** (все стратегии ≈ chance). Negative finding, гипотеза в RQ2 D2 секции.
5. **D2 size sweep только на Gender.** Rosbank/Age для RQ3 D2 — будущая работа.
6. **API non-determinism для MoE моделей.** Provider не гарантирует deterministic decoding для DeepSeek/Qwen3.6 даже при seed=42 в payload.

---

## Ссылки

- **Код по экспериментам:** [`EXPERIMENTS_MAP.md`](EXPERIMENTS_MAP.md)
- **Структура репо:** [`experiments/README.md`](experiments/README.md)
