# RAMD: Retrieval-Augmented Mutual Distillation

**Метод:** двухэтапный pipeline.

1. **Step 1** — OOF (out-of-fold) предсказания LLM teacher через 5-fold CV на train. Для каждого fold: kNN от других folds → CoT prompt с соседями → LLM soft prediction. Сохраняем в `results/ramd_openrouter/<dataset>_<teacher>_oof.npz`.
2. **Step 2** — CoLES retraining с reverse KL loss против teacher soft labels. Финальный AUC/accuracy в `results/ramd_kd/<dataset>_<teacher>_results.json`.

## Структура

```
ramd/
├── E1_4_gender_ramd_qwen7b.py       # E1.4 Gender, teacher = Qwen2.5-7B
├── E1_4_rosbank_ramd_qwen7b.py      # E1.4 Rosbank, teacher = Qwen2.5-7B
├── E1_5_gender_ramd_deepseek.py     # E1.5 Gender, teacher = DeepSeek-V3.2
├── E1_5_rosbank_ramd_deepseek.py    # E1.5 Rosbank, teacher = DeepSeek-V3.2
├── E1_5_age_ramd_deepseek.py        # E1.5 Age (4-class), teacher = DeepSeek-V3.2
│
├── step1_oof_binary.py              # shared step1 backend для binary tasks (gender + rosbank)
├── step1_oof_age.py                 # shared step1 backend для multiclass (age)
├── step2_distill.py                 # shared step2 backend (CoLES retraining с soft labels)
└── README.md
```

Per-dataset скрипты (`E1_*_ramd.py`) — тонкие wrappers через `distil.models.ramd.run_ramd_pipeline()`. Они задают `dataset_name` + `teacher_model_key` и вызывают backend через subprocess. Это позволяет:
- Вызывать backend напрямую через CLI: `python step1_oof_binary.py --datasets gender --models deepseek_v3`
- Или через wrapper: `python E1_5_gender_ramd.py` (полный pipeline + result.json)

## Запуск

```bash
# Из корня репо
python experiments/rq1_bidirectional/ramd/E1_5_gender_ramd_deepseek.py
```

## Зависимости

- CoLES baseline должен быть запущен заранее (нужны `embeddings/{dataset}/*_seed42.npy`)
- Установлен `OPENROUTER_API_KEY` для step1
- Бюджет: ~$0.30-1.50 на каждый dataset+teacher (зависит от модели)

## REPORT (single seed=42)

| # | Method | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|---|---|---|---|---|
| **E1.4** | Qwen2.5-7B teacher | 0.8630 | 0.8074 | — |
| **E1.5** | DeepSeek-V3.2 teacher | 0.8630 ± 0.0006 | 0.8072 ± 0.0034 | OOF running |

**Finding:** RAMD не выигрывает у простого LATTE (E1.2 Gender = 0.8674) и mutual_KL (E1.3 Rosbank = 0.8142). Iterative loop теоретически должен улучшать обе модели, но в практике CoLES почти насыщен (0.86 vs литературный потолок 0.89), а LLM ученик слабее CoLES учителя — soft labels не информативны. Negative result для discussion в paper.
