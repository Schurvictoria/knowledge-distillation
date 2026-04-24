# LATTE (standalone): LLM → CoLES

**Метод:** CoLES учится у LLM embeddings через contrastive loss. Веса LLM НЕ обновляются (односторонняя дистилляция).

## Результаты (val-split, seed=42, single training run)

| Dataset | CoLES baseline | LATTE α=0.1 | Δ |
|---|---|---|---|
| Gender | 0.8606 | **0.8713** | +1.07 пп |
| Rosbank | 0.8041 | **0.8082** | +0.41 пп |
| Age | 0.6283 | **0.6333** | +0.50 пп |