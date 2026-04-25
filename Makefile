PYTHON ?= python
EXPERIMENTS_DIR := experiments

.PHONY: help env install verify-env coles latte mutual-kl ramd rq1 rq2-d1 rq2-d2 rq3 aggregate clean-pyc

help:
	@echo "Targets:"
	@echo "  env             Create conda environment from environment.yml"
	@echo "  install         Install src/distil in editable mode"
	@echo "  verify-env      Check that data, embeddings, and ckpts are in place"
	@echo ""
	@echo "  coles           E1.1 — CoLES baselines on gender, rosbank, age"
	@echo "  latte           E1.2 — LATTE on 3 datasets (depends on coles)"
	@echo "  mutual-kl       E1.3 — LATTE + mutual KL on 3 datasets (depends on coles)"
	@echo "  ramd            E1.4 + E1.5 — RAMD step1 + step2 (depends on coles)"
	@echo "  rq1             All of E1.x"
	@echo ""
	@echo "  rq2-d1          E2.x — Teacher signal type"
	@echo "  rq2-d2          E3.x + E4.x — Prompt enrichment"
	@echo "  rq3             E5.3 + E6.1 — LLM size effect"
	@echo ""
	@echo "  aggregate       Generate REPORT_GENERATED.md from results/**/result.json"
	@echo "  clean-pyc       Remove __pycache__ and *.pyc"

env:
	conda env create -f environment.yml

install:
	pip install -e .

verify-env:
	$(PYTHON) scripts/verify_environment.py

coles:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/coles/run_gender_coles.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/coles/run_rosbank_coles.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/coles/run_age_coles.py

latte:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte/E1_2_gender_latte.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte/E1_2_rosbank_latte.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte/E1_2_age_latte.py

mutual-kl:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte_mutual_kl/E1_3_gender_mutual_kl.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte_mutual_kl/E1_3_age_mutual_kl.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/latte_mutual_kl/E1_3_all_mutual_kl.py

ramd:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_binary.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_age_4class.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq1_bidirectional/ramd/E1_4_E1_5_ramd_step2.py

rq1: coles latte mutual-kl ramd

rq2-d1:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d1_teacher_signals/response_based/E2_1_reverse_kl_seeds.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d1_teacher_signals/feature_based/E2_2_gender_llm4es.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d1_teacher_signals/feature_based/E2_2_rosbank_llm4es.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d1_teacher_signals/feature_based/E2_2_age_llm4es_v2.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d1_teacher_signals/combined/E2_4_combined_kd.py

rq2-d2:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d2_prompt_enrichment/by_enrichment_type/E3_3_E3_4_E3_5_gender_rosbank_cot_enrichments.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d2_prompt_enrichment/by_enrichment_type/E3_3_E3_4_E3_5_age_cot_enrichments.py
	$(PYTHON) $(EXPERIMENTS_DIR)/rq2_d2_prompt_enrichment/by_strategy/E4_1_E4_2_E4_3_strategy_matrix.py

rq3:
	$(PYTHON) $(EXPERIMENTS_DIR)/rq3_llm_size_effect/E5_x_extract_llm_embeddings.py \
		--model Qwen/Qwen2.5-3B-Instruct --teacher qwen25_3b
	$(PYTHON) $(EXPERIMENTS_DIR)/rq3_llm_size_effect/E6_1_gemma_runner.py

aggregate:
	$(PYTHON) scripts/aggregate_results.py

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -name '*.pyc' -delete
