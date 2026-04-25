# RAMD: Retrieval-Augmented Mutual Distillation

**Метод:** Итеративный loop:
1. CoLES → kNN соседи по CoLES embeddings
2. kNN подаются в LLM prompt как enrichment → LLM делает OOF предсказания
3. Soft labels из LLM возвращаются в CoLES через reverse KL
4. Повтор

## Скрипты

| Скрипт | Назначение |
|---|---|
| `run_ramd_kd.py` | Main RAMD loop (все 3 датасета) |
| `run_gender_ramd.py` | Gender-specific |
| `run_ramd_openrouter_oof.py` | Генерация OOF предсказаний от OpenRouter teacher (Step 1) |

## Результаты (Qwen2.5-7B teacher)

| Dataset | AUC/Acc |
|---|---|
| Gender | 0.8630 |
| Rosbank | 0.8074 |
| Age | — (не запускали) |

## TODO
- RAMD loop с **DeepSeek-V3.2** teacher — день 2 GPU training
- GPT-4o teacher — дропнут
