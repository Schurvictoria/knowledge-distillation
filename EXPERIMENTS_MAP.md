# Experiments → Code Map

Маппинг номеров экспериментов
---

## RQ1: Bidirectional vs Unidirectional

| # | Experiment | Script |
|---|---|---|
| **E1.1** | CoLES baseline | `experiments/rq1_bidirectional/coles/run_gender_coles.py`, `experiments/rq1_bidirectional/coles/run_rosbank_coles.py`, `experiments/rq1_bidirectional/coles/run_age_coles.py` |
| **E1.2** | LATTE (LLM → CoLES, contrastive, frozen LLM) | `experiments/rq1_bidirectional/latte/run_gender_latte.py`, `experiments/rq1_bidirectional/latte/run_rosbank_latte.py`, `experiments/rq1_bidirectional/latte/run_age_latte.py` |
| **E1.3** | LATTE + mutual KL (LLM ↔ CoLES) | `experiments/rq1_bidirectional/latte_mutual_kl/run_gender_true_bidirectional.py`, `run_all_true_bidirectional.py`, `run_age_bidir_fixed.py` |
| **E1.4** | RAMD (Qwen2.5-7B teacher) | `experiments/rq1_bidirectional/ramd/run_ramd_kd.py` (Step 2, local Qwen2.5-7B teacher) |
| **E1.5** | RAMD (DeepSeek-V3.2 teacher) | Step 1: `experiments/rq1_bidirectional/ramd/run_ramd_openrouter_oof.py --models deepseek_v3`<br>Step 2: `run_ramd_kd.py` с `RAMD_TEACHER=deepseek_v3` env var |
| **E1.6** | RAMD (GPT-4o) — dropped | — |

## RQ2 Direction 1: Teacher Signal Type

| # | Experiment | Script |
|---|---|---|
| **E2.1** | Response-based (reverse KL soft labels) | `experiments/rq2_d1_teacher_signals/response_based/run_blite_soft_distill.py`, `run_reverse_kl_seeds.py`, `run_gender_distill.py`, `run_gender_rkd.py` |
| **E2.2** | Feature-based (LLM4ES embedding concat) | `experiments/rq2_d1_teacher_signals/feature_based/run_gender_llm4es.py`, `run_rosbank_llm4es.py`, `run_age_llm4es.py`, `run_age_llm4es_v2.py` |
| **E2.3** | Relation-based (LATTE contrastive) — same as E1.2 | см. E1.2 |
| **E2.4** | All three combined (LATTE + mutual learning + LoRA) — same as E1.3 | см. E1.3 |

**Supporting (TAID/DAKD variants):**
- `experiments/rq2_d1_teacher_signals/relation_based/run_taid_crossmodal.py`
- `experiments/rq2_d1_teacher_signals/relation_based/run_taid_zscore_seeds.py`
- `experiments/rq2_d1_teacher_signals/relation_based/run_gender_taid_dakd.py`
- `experiments/rq2_d1_teacher_signals/relation_based/run_gender_contrastive_distill.py`
- `experiments/rq2_d1_teacher_signals/combined/run_combined_kd.py`

## RQ2 Direction 2: Enrichment Type (Qwen2.5-7B 4-bit, CoT)

| # | Experiment | Script |
|---|---|---|
| **E3.1** | None (baseline) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/run_no_enrich_zero.py` (baseline no-enrich) / `run_gender_rosbank_cot.py` (CoT+None) |
| **E3.2** | Prediction (XGB confidence) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/run_prediction_enrich.py` (Gender+Rosbank)<br>`run_prediction_enrich_age.py` (Age multiclass) |
| **E3.3** | Explanation (SHAP) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/run_gender_rosbank_cot.py` (mode=shap_cot)<br>`run_age_structured_cot.py` (Age) |
| **E3.4** | Retrieval (CoLES kNN) | `by_enrichment_type/run_gender_rosbank_cot.py` (mode=knn_cot)<br>`run_age_structured_cot.py` |
| **E3.5** | All combined (SHAP+kNN) | `by_enrichment_type/run_gender_rosbank_cot.py` (mode=both_cot)<br>`run_age_structured_cot.py` (full_cot) |

## RQ2 Direction 2: Strategy × Enrichment Matrix

| # | Experiment | Script |
|---|---|---|
| **E4.1** | Zero-shot × {None, SHAP, kNN, Both} | `experiments/rq2_d2_prompt_enrichment/by_strategy/run_strategy_matrix.py` |
| **E4.2** | Few-shot × {None, SHAP, kNN, Both} | `run_strategy_matrix.py` |
| **E4.3** | CoT × {None, SHAP, kNN, Both} | `run_strategy_matrix.py` + `run_cot_ablation.py` |

## RQ3 Direction 1: Teacher LLM Size Effect (LATTE)

| # | Experiment | Script |
|---|---|---|
| **E5.1** | Gemma 3n E2B — dropped (gated HF) | — |
| **E5.2** | Qwen2.5-7B — same as E1.2 | `experiments/rq1_bidirectional/latte/run_*_true_latte.py` |
| **E5.3** | Qwen2.5-3B proxy | `experiments/rq3_llm_size_effect/extract_llm_embeddings.py --model Qwen/Qwen2.5-3B-Instruct --teacher qwen25_3b` затем LATTE с этим файлом embeddings |
| **E5.4**-**E5.6** | Qwen3.6-35B / DeepSeek-R1 / GPT-4o — невозможно локально | — |

## RQ3 Direction 2: LLM Size Effect on Prompt Enrichment

| # | Experiment | Script |
|---|---|---|
| **E6.1** | Gemma 3-4B (Gender, no-enrich + kNN) | `experiments/rq3_llm_size_effect/run_gemma2b_knn.py --strategies zero_shot_none zero_shot_knn` |
| **E6.2** | Qwen2.5-7B (4-bit local) — same as E3.4 | см. RQ2 D2 |
| **E6.3** | Qwen3.6-35B-A3B (OpenRouter) | `experiments/rq2_d2_prompt_enrichment/by_model/run_qwen36_rosbank.py --datasets gender`<br>+ `by_enrichment_type/run_no_enrich_zero.py --models qwen36` |
| **E6.4** | DeepSeek-V3.2-Speciale (OpenRouter) | `by_model/run_deepseek_proper.py --datasets gender`<br>+ `run_no_enrich_zero.py --models deepseek_v3` |
| **E6.5** | GPT-4o — dropped | — |

## CoT Reasoning Effect (thinking on/off)

| # | Experiment | Script / Artifact |
|---|---|---|
| **E7.1** | Qwen3.6-35B thinking on/off | Old results in `results/openrouter/gender_qwen36_35b_thinking_{on,off}_knn.json` |
| **E7.2** | DeepSeek-V3.2 thinking=on only (off not supported by API) | `by_model/run_deepseek_proper.py` |
| **E7.3** | GLM-4.7 bonus ablation | `by_model/run_glm_gold_cot.py`, `run_glm_proper_cot.py` (proper_cot_off/on files) |

---

## Infrastructure / Data Pipeline

| Purpose | Script |
|---|---|
| CoLES baseline training | `experiments/infrastructure/run_{gender,rosbank,age}_coles.py` |
| Embeddings saved at | `embeddings/{dataset}/*_seed42.npy` |
| LLM embedding extractor (generic, for new teachers) | `experiments/rq3_llm_size_effect/extract_llm_embeddings.py` |
| Multi-seed LGBM eval wrapper | `experiments/infrastructure/run_seeded_eval.py` |
| Shared OpenRouter utilities (imported by ALL OR-based scripts) | `run_openrouter_experiments.py` (в корне) |

## Reproducibility notes

- **Train/test split:** все скрипты используют `embeddings/{dataset}/*_seed42.npy` (seed=42, stratified).
- **LGBM random_state=42** везде.
- **API calls:** `seed=42` в OpenRouter payload (новые скрипты). Старые `*` в REPORT — без API seed.
- **PyTorch training:** LATTE/mutual_KL использует `torch.manual_seed(42) + pl.seed_everything(42) + cudnn.deterministic=True` (после фикса `db9506c`).
- **RAMD Step 2:** `SEEDS=[42,123,456,789,1024]`, `set_seed(s)` для каждого.
