# Per-Model Runners

Основные runners для каждой LLM + датасета. Запуск из корня репо.

## Скрипты по моделям

| Model | Runner | Notes |
|---|---|---|
| **DeepSeek-V3.2-Speciale** | `run_deepseek_proper.py` | max_tokens=8192, temp=0.6, reasoning=on |
| **Qwen3.6-Plus (35B MoE)** | `run_qwen36_rosbank.py` | max_tokens=4096, temp=0.6, reasoning=on. Принимает `--datasets gender rosbank` |
| **GLM-4.7** | `run_glm_fewshot_proper.py`, `run_glm_fewshot_random.py` | temp=0, reasoning=off (детерминировано) |
| **GLM-4.7 Age 4-class** | `run_glm_age.py` | multiclass output schema. Env: `AGE_MODEL_ID`, `AGE_MODEL_SHORT` чтобы сменить модель (Qwen3.6, DeepSeek) |
| **Qwen2.5-7B-Instruct (local 4-bit)** | `run_gender_llm.py`, `run_rosbank_llm.py`, `run_age_llm.py` | GPU inference через HF transformers |

## Ранние эксперименты (historical)

| `run_glm_proper_cot.py`, `run_glm_gold_cot.py` | GLM CoT ablation, предыдущие версии |

## Стратегии (4 для binary / 2 для Age)

- `zero_shot_knn` — zero-shot + enrichment текстом "Similar clients: X pos, Y neg"
- `few_shot_random` — baseline: 2 random demos (1 per class)
- `few_shot_knn` — 4 kNN-retrieved demos (2 pos + 2 neg, balanced)
- `few_shot_cot_knn` — kNN demos с chain-of-thought reasoning в assistant message

## Пример запуска

```bash
cd /workspace/repos/knowledge-distillation

# DeepSeek Gender с seed=42
python experiments/rq2_d2_prompt_enrichment/by_model/run_deepseek_proper.py \
    --datasets gender --budget 5.0

# Qwen3.6 Rosbank
python experiments/rq2_d2_prompt_enrichment/by_model/run_qwen36_rosbank.py \
    --datasets rosbank --budget 3.0
```

Cache → `results/openrouter/{dataset}_{model}_{strategy}.json`
