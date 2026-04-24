# LATTE (standalone): LLM → CoLES

**Метод:** CoLES учится у LLM embeddings через contrastive loss. Веса LLM НЕ обновляются (односторонняя дистилляция).

## Скрипты

| Скрипт | Датасет | Назначение |
|---|---|---|
| `run_gender_true_latte.py` | Gender | Main LATTE на Gender |
| `run_rosbank_true_latte.py` | Rosbank | Main LATTE на Rosbank |
| `run_age_true_latte.py` | Age | Main LATTE на Age |
| `run_gender_latte_distill.py` | Gender | Variant: с learnable temperature |
| `run_gender_latte_alpha005.py` | Gender | Ablation: alpha=0.05 для contrastive loss |

## Результаты

| Dataset | AUC/Acc |
|---|---|
| Gender | 0.8674 |
| Rosbank | 0.8057 |
| Age | 0.6429 |

Δ vs CoLES baseline: Gender +0.5 пп, Rosbank +0.03 пп, Age +0.84 пп.
