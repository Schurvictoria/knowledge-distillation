# Experiments → Code Map

Маппинг номеров экспериментов из [`REPORT.md`](REPORT.md) к исходникам.
Все скрипты запускаются **из корня репо**.

---

## RQ1: Bidirectional vs Unidirectional

| # | Experiment | Script |
|---|---|---|
| **E1.1** | CoLES baseline | `experiments/rq1_bidirectional/coles/run_{gender,rosbank,age}_coles.py` |
| **E1.2** | LATTE (LLM → CoLES, contrastive, frozen LLM) | `experiments/rq1_bidirectional/latte/E1_2_{gender,rosbank,age}_latte.py` |
| **E1.3** | LATTE + mutual KL (LLM ↔ CoLES) | `experiments/rq1_bidirectional/latte_mutual_kl/E1_3_{gender,age,all}_mutual_kl.py` |
| **E1.4** | RAMD (Qwen2.5-7B teacher) | `experiments/rq1_bidirectional/ramd/E1_4_E1_5_ramd_step2.py` (local Qwen2.5-7B teacher) + `E1_4_gender_ramd_alt.py` (alternative) |
| **E1.5** | RAMD (DeepSeek-V3.2 teacher) | Step 1: `experiments/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_{binary,age_4class}.py --models deepseek_v3`<br>Step 2: `E1_4_E1_5_ramd_step2.py` с `RAMD_TEACHER=deepseek_v3` |

**Ablations:**
- `experiments/rq1_bidirectional/latte/ablations/E1_2_gender_latte_alpha_ablation.py` — α sweep для E1.2 (обоснование best α=0.1)
- `experiments/rq1_bidirectional/latte/ablations/E1_2_gender_latte_variant.py` — variant методов (LR, dropout)
- `experiments/rq1_bidirectional/latte_mutual_kl/ablations/E1_3_gender_mutual_kl_early.py` — early-stop ablation для E1.3

## RQ2 Direction 1: Teacher Signal Type

| # | Experiment | Script |
|---|---|---|
| **E2.1** | Response-based (reverse KL soft labels) | `experiments/rq2_d1_teacher_signals/response_based/E2_1_{blite_soft_distill,reverse_kl_seeds,gender_distill,gender_rkd}.py` |
| **E2.2** | Feature-based (LLM4ES embedding concat) | `experiments/rq2_d1_teacher_signals/feature_based/E2_2_{gender,rosbank,age}_llm4es.py` + `E2_2_age_llm4es_v2.py` (improved 5-epoch run) |
| **E2.3** | Relation-based (LATTE contrastive) — same as E1.2 | см. E1.2 |
| **E2.4** | All three combined (soft labels + embeddings + contrastive) | `experiments/rq2_d1_teacher_signals/combined/E2_4_combined_kd.py` |

## RQ2 Direction 2: Enrichment Type (Qwen2.5-7B 4-bit, CoT)

| # | Experiment | Script |
|---|---|---|
| **E3.1** | None (baseline) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/E3_1_no_enrich.py` |
| **E3.2** | Prediction (XGB confidence) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/E3_2_prediction_enrich.py` (Gender+Rosbank)<br>`E3_2_prediction_enrich_age.py` (Age multiclass) |
| **E3.3** | Explanation (SHAP) | `experiments/rq2_d2_prompt_enrichment/by_enrichment_type/E3_3_E3_4_E3_5_gender_rosbank_cot_enrichments.py` (mode=shap_cot)<br>`E3_3_E3_4_E3_5_age_cot_enrichments.py` (Age) |
| **E3.4** | Retrieval (CoLES kNN) | те же файлы (mode=knn_cot) |
| **E3.5** | All combined (SHAP+kNN) | те же файлы (mode=both_cot / full_cot) |

**Local Qwen2.5-7B runners (для E3.x на 3 датасетах):**
- `experiments/rq2_d2_prompt_enrichment/by_model/E3_x_{gender,rosbank,age}_llm_local.py`

## RQ2 Direction 2: Strategy × Enrichment Matrix

| # | Experiment | Script |
|---|---|---|
| **E4.1** | Zero-shot × {None, SHAP, kNN, Both} | `experiments/rq2_d2_prompt_enrichment/by_strategy/E4_1_E4_2_E4_3_strategy_matrix.py` |
| **E4.2** | Few-shot × {None, SHAP, kNN, Both} | тот же `E4_1_E4_2_E4_3_strategy_matrix.py` |
| **E4.3** | CoT × {None, SHAP, kNN, Both} | `E4_1_E4_2_E4_3_strategy_matrix.py` + `E4_3_cot_ablation.py` |

## RQ3 Direction 1: Teacher LLM Size Effect (LATTE)

| # | Experiment | Script |
|---|---|---|
| **E5.0** | Qwen2.5-0.5B-Instruct (Gender ladder) | `experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_0_gender_latte_qwen0_5b.py` *(pending)* |
| **E5.1** | Qwen2.5-1.5B-Instruct (Gender ladder) | `experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_1_gender_latte_qwen1_5b.py` *(pending)* |
| **E5.2** | Qwen2.5-3B-base (existing canonical, used in E1.2) | `experiments/rq1_bidirectional/latte/E1_2_*_latte.py` |
| **E5.2-Instruct** | Qwen2.5-3B-Instruct (Gender ladder) | `experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_2_gender_latte_qwen3b_instruct.py` *(pending)* |
| **E5.3** | Qwen2.5-7B-Instruct (Gender ladder, top of ladder) | `experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_3_gender_latte_qwen7b.py` *(pending)* |

## RQ3 Direction 2: LLM Size Effect on Prompt Enrichment

| # | Experiment | Script |
|---|---|---|
| **E6.1** | Gemma 3-4B (Gender, no-enrich + kNN) | `experiments/rq3_llm_size_effect/E6_1_gemma_runner.py --strategies zero_shot_none zero_shot_knn` |
| **E6.2** | Qwen2.5-7B (4-bit local) — same as E3.4 | см. RQ2 D2 |
| **E6.3** | Qwen3.6-35B-A3B (OpenRouter) | `experiments/rq2_d2_prompt_enrichment/by_model/E6_3_qwen36_runner.py --datasets gender` |
| **E6.4** | DeepSeek-V3.2-Speciale (OpenRouter) | `experiments/rq2_d2_prompt_enrichment/by_model/E6_4_deepseek_runner.py --datasets gender` |

**GLM-4.7 helpers (E3.x для GLM):**
- `experiments/rq2_d2_prompt_enrichment/by_model/E3_x_glm_{fewshot_proper,fewshot_random,age_runner}.py`

`E3_x_glm_fewshot_proper.py` содержит shared `build_cot_reasoning` — импортируется из `E6_3_qwen36_runner.py` и `E6_4_deepseek_runner.py`.

## CoT Reasoning Effect (thinking on/off)

| # | Experiment | Script / Artifact |
|---|---|---|
| **E7.1** | Qwen3.6-35B thinking on/off | Old results in `results/openrouter/gender_qwen36_35b_thinking_{on,off}_knn.json` |
| **E7.2** | DeepSeek-V3.2 thinking=on only (off not supported by API) | `experiments/rq2_d2_prompt_enrichment/by_model/E6_4_deepseek_runner.py` |
| **E7.3** | GLM-4.7 bonus ablation | `experiments/rq2_d2_prompt_enrichment/by_model/E7_3_glm_{gold,proper}_cot.py` |

---

## Shared Infrastructure

| Purpose | Script |
|---|---|
| CoLES baseline training (E1.1) | `experiments/rq1_bidirectional/coles/run_{gender,rosbank,age}_coles.py` |
| Embeddings saved at | `embeddings/{dataset}/{emb_train,emb_test,cids_train,cids_test,y_train,y_test}_seed42.npy` |
| LLM embedding extractor (generic, used in E5.3) | `experiments/rq3_llm_size_effect/E5_x_extract_llm_embeddings.py` |
| Shared OpenRouter utilities (imported by ALL OR-based scripts) | `run_openrouter_experiments.py` (в корне репо) |

**Disabled (зарезервировано для будущего):**
- `experiments/shared/run_seeded_eval.py` — multi-seed LGBM eval wrapper (был для расчёта ±std из существующих чекпоинтов)
- `experiments/shared/run_age_supcon_latte.py` — SupCon variant LATTE для multiclass (Age=0.644 в FINAL_RESULTS, не в REPORT)

## Reproducibility notes

- **Train/test split:** все скрипты используют `train_test_split(test_size=0.1, random_state=42, stratify=targets)`. Embeddings: `embeddings/{dataset}/*_seed42.npy`.
- **PyTorch training:** все скрипты теперь имеют полный seed-блок (`random + np.random + torch + cuda + pl.seed_everything(workers=True) + PYTHONHASHSEED + cudnn.deterministic + cudnn.benchmark=False`).
- **OpenRouter API:** `"seed": 42` в payload, `"temperature": 0` для детерминированных моделей, `0.6` для reasoning (DeepSeek/Qwen3.6), `0.7` для self-consistency (E7.3 gold_cot).
- **LGBM:** `random_state=42` для single-seed runs; `[42,123,456,789,1024]` для multi-seed eval.
- **kNN:** `NearestNeighbors(n_neighbors=10, metric="cosine")` везде.
- **Honest model selection:** для всех LATTE/Bidirectional/RAMD скриптов test читается **только** для финального reporting на эпохе с лучшим val.
- **LLM4ES embedding pooling:** mean over last 8 hidden layers, then masked mean over sequence (E5_x теперь следует той же конвенции что и E2_2 — fixed 2026-04-25).
