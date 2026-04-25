# LATTE + mutual KL: LLM ↔ CoLES

**Метод:** LATTE contrastive loss + **mutual KL divergence** между softmax output'ами LLM и CoLES. **Оба** обновляют веса одновременно (двусторонняя дистилляция).

## Скрипты

| Скрипт | Датасет |
|---|---|
| `run_gender_true_bidirectional.py` | Gender |
| `run_gender_bidirectional.py` | Gender (ранняя версия) |
| `run_age_bidir_fixed.py` | Age (с баг-фиксом) |
| `run_all_true_bidirectional.py` | all 3 datasets в одном скрипте |

## Результаты (5 seeds mean)

| Dataset | AUC/Acc |
|---|---|
| Gender | 0.8676 |
| Rosbank | 0.8142 |
| Age | 0.6363 |

**Находка:** mutual KL помогает на Rosbank (+0.85 пп vs LATTE) но немного проседает на Age (−0.66 пп).
