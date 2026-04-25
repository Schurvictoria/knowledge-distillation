# RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

**Вопрос:** Какой тип учительского сигнала эффективнее?

## Подпапки по типам сигнала

| Папка | Сигнал | Что получает student | Gender | Rosbank | Age |
|---|---|---|---|---|---|
| [`response_based/`](response_based/) | Soft labels (Reverse KL) | "male 73%" | 0.8633 | 0.8074 | 0.6399 |
| [`feature_based/`](feature_based/) | LLM embeddings (2048d) | LLM4ES concat → LGBM | 0.864 | 0.819 | 0.640 |
| `relation_based/` | Contrastive alignment | "A похож на B" | — | — | — |
| [`combined/`](combined/) | Soft + emb + contrastive | All three | 0.8676 | 0.8142 | 0.6363 |

**Примечание:** Relation-based (E2.3) — то же что LATTE (E1.2), смотри [`../rq1_bidirectional/latte/`](../rq1_bidirectional/latte/). Результаты: 0.8674/0.8057/0.6429.

## Структура per-dataset

```
response_based/                              # E2.1
├── E2_1_gender_reverse_kl.py                # wrapper для Gender
├── E2_1_rosbank_reverse_kl.py
├── E2_1_age_reverse_kl.py
└── reverse_kl_distill_backend.py            # shared backend, 5 seeds

feature_based/                               # E2.2 — см. локальный README
├── E2_2_gender_llm4es.py
├── E2_2_rosbank_llm4es.py
├── E2_2_age_llm4es.py
└── README.md

combined/                                    # E2.4
├── E2_4_gender_combined.py                  # wrapper для Gender
├── E2_4_rosbank_combined.py
├── E2_4_age_combined.py
└── combined_kd_backend.py                   # shared backend
```

Per-dataset wrappers — тонкие subprocess-launchers к backend. Можно запускать как wrapper (`python E2_1_gender_reverse_kl.py`) или backend напрямую с CLI args (`python reverse_kl_distill_backend.py --datasets gender age`).
