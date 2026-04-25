# RQ3: LLM Size Effect

**Вопрос:** влияет ли размер LLM (а) на качество дистилляции в structured модель (D1) и (б) на эффективность kNN обогащения промптов (D2)?

## Структура

```
rq3_llm_size_effect/
├── d1_teacher_size_for_latte/                    # D1 — размер LLM teacher для LATTE
│   └── E5_x_extract_llm_embeddings.py            # generic LLM embeddings extractor
│
└── d2_size_for_enrichment/                       # D2 — размер LLM для kNN-CoT enrichment
    ├── gemma_3_4b/                               # E6.1 — Gemma 3-4B (4B, local 4-bit)
    │   └── E6_1_gender_gemma.py
    ├── qwen_25_7b/                               # E6.2 — Qwen2.5-7B (7B, local 4-bit)
    │   ├── E6_2_gender_qwen7b.py
    │   ├── E6_2_rosbank_qwen7b.py
    │   └── E6_2_age_qwen7b.py
    ├── qwen_36_35b/                              # E6.3 — Qwen3.6-35B-A3B (OpenRouter)
    │   └── E6_3_gender_qwen36.py
    ├── deepseek_v32/                             # E6.4 — DeepSeek-V3.2-Speciale (OpenRouter)
    │   └── E6_4_gender_deepseek.py
    └── glm_4_7/                                  # E7.3 — GLM-4.7 (thinking off, temp=0)
        └── E7_3_gender_glm_proper.py
```

## Direction 1 (LLM → Structured) — E5.x

LATTE с разными teacher LLMs.

| # | Teacher LLM | Size | Скрипт | Статус |
|---|---|---|---|---|
| **E5.1** | Gemma 3n E2B | 2B | — | dropped (gated на HF) |
| **E5.2** | Qwen2.5-7B-Instruct | 7B | `../rq1_bidirectional/latte/E1_2_*.py` | ✓ done = E1.2 |
| **E5.3** | Qwen2.5-3B (proxy) | 3B | `d1_teacher_size_for_latte/E5_x_extract_llm_embeddings.py --model Qwen/Qwen2.5-3B-Instruct --teacher qwen25_3b` затем LATTE retrain | pending |
| **E5.4-E5.6** | Qwen3.6-35B / DeepSeek-R1 / GPT-4o | 35B+ | — | impossible локально |

## Direction 2 (Structured → LLM, kNN CoT) — E6.x

Только Gender, single seed=42.

| # | LLM | Size | Скрипт | No enrich | + kNN | Δ |
|---|---|---|---|---|---|---|
| **E6.1** | Gemma 3-4B | 4B local | `gemma_3_4b/E6_1_gender_gemma.py` | 0.5280 | 0.7669 | +23.9 pp |
| **E6.2** | Qwen2.5-7B-Instruct | 7B local | `qwen_25_7b/E6_2_gender_qwen7b.py` | 0.498 | 0.762 | +26 pp |
| **E6.3** | Qwen3.6-35B-A3B | 35B MoE | `qwen_36_35b/E6_3_gender_qwen36.py` | 0.5077 | 0.7790 | +27.1 pp |
| **E6.4** | DeepSeek-V3.2-Speciale | 671B MoE | `deepseek_v32/E6_4_gender_deepseek.py` | 0.5152 | 0.7828 | +26.8 pp |
| **E6.5** | GPT-4o | ~200B | — | dropped (closed) | — | — |

**Главный finding:** kNN даёт **+25-29 pp AUC независимо от размера модели** (0.5 → 0.77-0.78). Размер LLM не bottleneck.

## CoT Reasoning Effect — E7.x (только Gender)

Влияет ли thinking on/off на качество?

| # | LLM | Size | Скрипт | Off | On | Δ |
|---|---|---|---|---|---|---|
| **E7.1** | Qwen3.6-35B | 35B MoE | results in `results/openrouter/` | 0.7138 | 0.7151 | +0.13 pp |
| **E7.2** | DeepSeek-V3.2 | 671B MoE | `deepseek_v32/E6_4_gender_deepseek.py` (thinking always on) | N/A | 0.7828 | — |
| **E7.3** | GLM-4.7 | ~9B | `glm_4_7/E7_3_gender_glm_proper.py` | 0.7712 | — | — |

## Заметка про E6.2 = E3.x

Qwen2.5-7B local — тот же canonical 4-bit inference что используется в **REPORT.md E3.x таблице** (RQ2 D2). Один скрипт = два аспекта анализа.

## Пример запуска

```bash
# E5.3 — извлечение Qwen2.5-3B embeddings для LATTE retrain
python experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_x_extract_llm_embeddings.py \
    --model Qwen/Qwen2.5-3B-Instruct --teacher qwen25_3b --datasets gender rosbank age

# E6.4 — DeepSeek на Gender
python experiments/rq3_llm_size_effect/d2_size_for_enrichment/deepseek_v32/E6_4_gender_deepseek.py \
    --datasets gender --budget 5.0
```
