# RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

**Вопрос:** Какой тип учительского сигнала эффективнее?

## Подпапки по типам сигнала

| Папка | Сигнал | Что получает student | Gender | Rosbank | Age |
|---|---|---|---|---|---|
| [`response_based/`](response_based/) | Soft labels | "male 73%" | 0.8633 | 0.8074 | 0.6399 |
| [`feature_based/`](feature_based/) | LLM embeddings (2048d) | LLM4ES concat → LGBM | 0.864 | 0.819 | 0.640 |
| [`relation_based/`](relation_based/) | Contrastive alignment (TAID/DAKD variants) | "A похож на B" | — | — | — |
| [`combined/`](combined/) | Soft + emb + contrastive | All three | 0.8676 | 0.8142 | 0.6363 |

**Примечание:** Relation-based (LATTE без mutual KL) описан в [`../rq1_bidirectional/latte/`](../rq1_bidirectional/latte/) — его результат 0.8674/0.8057/0.6429.

**Статус:** 5 seeds × 3 датасета — закрыто.
