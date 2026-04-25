# RQ3: LLM Size Effect

## Direction 1: LLM → Structured (True LATTE distillation)

**Вопрос:** Влияет ли размер LLM-teacher на качество дистилляции?

**Method:** True LATTE (contrastive alignment). Student: CoLES → LightGBM.

| Teacher LLM | Script | Статус |
|---|---|---|
| Gemma 3n E2B (2B) | — | не делали (дропнуто) |
| Qwen2.5-7B-Instruct (7B) | `run_gender_true_latte.py`, etc. | ✓ готово |
| Qwen3.6-35B-A3B (35B MoE) | нужен новый LATTE train с Qwen3.6 embeddings | TODO GPU |
| DeepSeek-R1 (671B MoE) | нужен LATTE train с DS embeddings | TODO GPU |
| GPT-4o (~200B) | — | дропнуто |

---

## Direction 2: Structured → LLM (kNN enrichment size effect)

**Вопрос:** Влияет ли размер LLM на эффективность kNN обогащения промптов?

**Method:** Zero-shot + kNN. Dataset: Gender.

| LLM | Size | Script (корень репо) | No enrichment | + kNN | Δ |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | `run_gender_rosbank_cot.py` | 0.498 | 0.762 | +26 pp |
| GLM-4.7 | ~9B | `run_glm_fewshot_proper.py`, `run_no_enrich_zero.py` | 0.5140 | 0.7712 | +25.7 pp |
| Qwen3.6-Plus | 35B MoE | `run_qwen36_rosbank.py`, `run_no_enrich_zero.py` | 0.4934* | 0.7834* | +29.0 pp |
| DeepSeek-V3.2-Speciale | 671B MoE | `run_deepseek_proper.py`, `run_no_enrich_zero.py` | 0.5152* | 0.7828* | +26.8 pp |

`*` = без API seed=42 (variance ±1-2 пп).

**Паттерн:** kNN даёт +25-29 pp AUC независимо от размера модели (0.5 → 0.77-0.78).

## TODO (RQ3 D1 — GPU training, день 2)
- LATTE retrain с Qwen3.6-35B teacher embeddings
- LATTE retrain с DeepSeek teacher embeddings
