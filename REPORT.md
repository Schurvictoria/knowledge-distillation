## RQ1: Bidirectional vs Unidirectional

| Method | Type | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|--------|------|-------------|---------------|-----------|
| CoLES baseline | No transfer | 0.8626 | 0.8054 | 0.6345 |
| True LATTE | LLM→Struct only | 0.8674 | 0.8057 | 0.6429 |
| True Bidirectional | Both (joint) | 0.8676 | 0.8142 | 0.6363 |
| kNN CoT → LLM | Struct→LLM only | — | — | — |
| RAMD (weak teacher) | Bidirectional loop | 0.8630 | 0.8074 | — |
| RAMD (strong teacher) | Bidirectional loop | ? | ? | ? |

## RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

| Signal Type | Method | Gender | Rosbank | Age |
|------------|--------|--------|---------|-----|
| Response-based (soft labels) | Reverse KL | 0.8633 | 0.8074 | 0.6399 |
| Feature-based (embeddings) | LLM4ES concat | 0.864 | 0.819 | 0.640 |
| Relation-based (contrastive) | True LATTE | 0.8674 | 0.8057 | 0.6429 |
| All combined | True Bidirectional | 0.8676 | 0.8142 | 0.6363 |

## RQ2 Direction 2: Enrichment Type (Structured → LLM)

| Enrichment | Structured Model | Gender | Rosbank | Age |
|-----------|-----------------|--------|---------|-----|
| None | — | 0.498 | 0.499 | 0.249 |
| Prediction | XGBoost confidence | ? | ? | ? |
| Explanation | XGBoost SHAP | 0.606 | 0.637 | ? |
| Retrieval | CoLES kNN | 0.762 | 0.766 | 0.250 |
| All combined | XGBoost + CoLES | 0.745 | 0.751 | ? |

## RQ2 Direction 2: Strategy × Enrichment (матрица)

| Strategy | None | + SHAP | + kNN | + Both |
|----------|------|--------|-------|--------|
| Zero-shot | 0.498 | ? | ? | ? |
| Few-shot | 0.578 | ? | ? | ? |
| CoT | ? | 0.606 | 0.762 | 0.745 |

## RQ3: LLM Size Effect

### Direction 1 (LLM → Structured Models, True LATTE distillation)

| Teacher LLM | Size | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|-------------|------|-------------|---------------|-----------|
| Gemma 3n E2B | 2B | ? | ? | ? |
| Qwen2.5-7B-Instruct | 7B | 0.8674 | 0.8057 | 0.6429 |
| Qwen3.6-35B-A3B | 35B MoE | ? | ? | ? |
| DeepSeek-R1-0528 | 671B MoE | ? | ? | ? |
| GPT-4o | ~200B | ? | ? | ? |

### Direction 2 (Structured Models → LLM, kNN CoT enrichment)

| LLM | Size | No enrichment | + kNN CoT | Δ |
|-----|------|--------------|-----------|---|
| Gemma 3n E2B | 2B | ? | ? | ? |
| Qwen2.5-7B-Instruct | 7B | 0.498 | 0.762 | +26 pp |
| Qwen3.6-35B-A3B | 35B MoE | ? | ? | ? |
| DeepSeek-R1-0528 | 671B MoE | ? | ? | ? |
| GPT-4o | ~200B | ? | ? | ? |

### CoT Reasoning Effect

| Teacher LLM | Size | Thinking=off | Thinking=on | Δ |
|-------------|------|-------------|-------------|---|
| Qwen3.6-35B-A3B | 35B MoE | ? | ? | ? |
| DeepSeek-R1-0528 | 671B MoE | ? | ? | ? |
