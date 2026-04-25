# Results

Каждая подпапка соответствует одному эксперименту. Маппинг с `EXPERIMENTS_MAP.md`:

## RQ1 — Bidirectional vs Unidirectional

| Папка | Эксперимент | Скрипт |
|---|---|---|
| `gender_coles/`, `rosbank_coles/`, `age_coles/` | E1.1 CoLES baseline | `experiments/rq1_bidirectional/coles/run_*_coles.py` |
| `gender_true_latte/`, `rosbank_true_latte/`, `age_true_latte/` | E1.2 LATTE | `experiments/rq1_bidirectional/latte/E1_2_*_latte.py` |
| `gender_true_bidirectional/`, `rosbank_true_bidirectional/`, `age_true_bidirectional/` | E1.3 LATTE + mutual KL | `experiments/rq1_bidirectional/latte_mutual_kl/E1_3_*_mutual_kl.py` |
| `ramd_openrouter/` | E1.4/E1.5 RAMD step 1 (OOF от LLM teachers) | `experiments/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_*.py` |
| `ramd_kd/` | E1.4/E1.5 RAMD step 2 (CoLES retraining) | `experiments/rq1_bidirectional/ramd/E1_4_E1_5_ramd_step2.py` |

## RQ2 D1 — Teacher signal type

| Папка | Эксперимент | Скрипт |
|---|---|---|
| `gender_distillation/`, `gender_rkd/` | E2.1 response-based (Hinton soft labels, RKD) | `experiments/rq2_d1_teacher_signals/response_based/E2_1_gender_*.py` |
| `blite_soft_distill/` | E2.1 b-lite soft distillation | `E2_1_blite_soft_distill.py` |
| `reverse_kl/` | E2.1 Reverse KL (5 seeds) | `E2_1_reverse_kl_seeds.py` |
| `gender_llm4es/`, `rosbank_llm4es/`, `age_llm4es/` | E2.2 LLM4ES feature distillation | `experiments/rq2_d1_teacher_signals/feature_based/E2_2_*_llm4es.py` |
| `combined_kd/` | E2.4 All three signals combined | `experiments/rq2_d1_teacher_signals/combined/E2_4_combined_kd.py` |

## RQ2 D2 — Prompt enrichment

| Папка | Эксперимент | Скрипт |
|---|---|---|
| `openrouter/` | E3.x / E4.x / E6.x / E7.x — все OpenRouter API runs | `experiments/rq2_d2_prompt_enrichment/by_*` |

(Для local Qwen 7B запусков пути зависят от конкретного скрипта — обычно тоже в `openrouter/` или в результатах самого скрипта.)

## RQ3 — LLM size effect

| Папка | Эксперимент | Скрипт |
|---|---|---|
| `gender_qwen25_3b_llm_embeddings/`, `rosbank_qwen25_3b_llm_embeddings/`, `age_qwen25_3b_llm_embeddings/` | E5.3 — извлечение Qwen2.5-3B embeddings для LATTE retrain | `experiments/rq3_llm_size_effect/E5_x_extract_llm_embeddings.py` |

## Поддерживающие папки

| Папка | Назначение |
|---|---|
| `gender_latte_alpha_ablation/`, `gender_latte_variant/`, `gender_mutual_kl_early/` | Ablations к E1.2/E1.3 (обоснование best α, early stopping и т.д.) |
| `seeded_eval/` | Multi-seed LGBM eval wrapper (currently disabled in `experiments/shared/run_seeded_eval.py`) |

## Что хранится в каждой папке

- `result.json` — стандартизированная схема (metrics + config + git_commit + runtime). Коммитится в git.
- `*.pt` — PyTorch чекпоинты (gitignored, большие файлы)
- `*.npy`, `*.npz` — pre-computed embeddings/predictions (gitignored)
- `*.csv` — per-seed детали (если есть)

Структура `result.json`:
```json
{
  "experiment_id": "E1_2_gender",
  "rq": "RQ1",
  "method": "LATTE",
  "dataset": "gender",
  "task_type": "binary",
  "metrics": {"roc_auc": 0.8674, "accuracy": 0.78, "f1": 0.74},
  "config": {"alpha": 0.1, "learning_rate": 5e-4, "epochs": 10, "seed": 42},
  "git_commit": "abc123def",
  "torch_version": "2.1.2",
  "ptls_version": "0.7.0",
  "runtime_seconds": 7842,
  "timestamp_utc": "2026-04-25T03:15:00Z",
  "artifacts": {"checkpoint": "results/gender_true_latte/coles_finetuned_alpha0.1.pt"}
}
```
