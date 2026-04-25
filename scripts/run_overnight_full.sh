#!/usr/bin/env bash
# Overnight execution of GPU-bound experiments in correct dependency order.
# Idempotent: skips experiments whose result.json already exists.
# Run from repository root.
#
# Usage:
#   bash scripts/run_overnight_full.sh
#   tmux new -s overnight 'bash scripts/run_overnight_full.sh'

set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

LOG_DIRECTORY="$REPOSITORY_ROOT/logs"
mkdir -p "$LOG_DIRECTORY"

OVERALL_LOG="$LOG_DIRECTORY/overnight_full_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$OVERALL_LOG") 2>&1

echo "=================================================================="
echo "Overnight run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repository: $REPOSITORY_ROOT"
echo "Log file:   $OVERALL_LOG"
echo "=================================================================="

run_experiment() {
    local experiment_id="$1"
    local script_path="$2"
    shift 2
    local extra_arguments=("$@")

    local result_file="results/${experiment_id}/result.json"
    if [[ -f "$result_file" ]]; then
        echo "[SKIP] $experiment_id — result.json already exists at $result_file"
        return 0
    fi

    if [[ ! -f "$script_path" ]]; then
        echo "[ERROR] $experiment_id — script not found: $script_path"
        return 1
    fi

    local start_timestamp
    start_timestamp=$(date +%s)
    echo "[START] $experiment_id at $(date -u +%H:%M:%S) — $script_path ${extra_arguments[*]}"

    if python "$script_path" "${extra_arguments[@]}"; then
        local elapsed=$(($(date +%s) - start_timestamp))
        echo "[DONE]  $experiment_id in ${elapsed}s"
    else
        local exit_code=$?
        echo "[FAIL]  $experiment_id with exit code $exit_code"
        return "$exit_code"
    fi
}

echo ""
echo "--- Phase 1: CoLES baselines (E1.1) ---"
run_experiment "E1_1_gender" experiments/rq1_bidirectional/coles/run_gender_coles.py || true
run_experiment "E1_1_rosbank" experiments/rq1_bidirectional/coles/run_rosbank_coles.py || true
run_experiment "E1_1_age" experiments/rq1_bidirectional/coles/run_age_coles.py || true

echo ""
echo "--- Phase 2: LATTE (E1.2) ---"
run_experiment "E1_2_gender" experiments/rq1_bidirectional/latte/E1_2_gender_latte.py || true
run_experiment "E1_2_rosbank" experiments/rq1_bidirectional/latte/E1_2_rosbank_latte.py || true
run_experiment "E1_2_age" experiments/rq1_bidirectional/latte/E1_2_age_latte.py || true

echo ""
echo "--- Phase 3: LATTE + mutual KL (E1.3) ---"
run_experiment "E1_3_gender" experiments/rq1_bidirectional/latte_mutual_kl/E1_3_gender_mutual_kl.py || true
run_experiment "E1_3_age" experiments/rq1_bidirectional/latte_mutual_kl/E1_3_age_mutual_kl.py || true

echo ""
echo "--- Phase 4: RAMD (E1.4 + E1.5) ---"
run_experiment "E1_5_step1_binary" experiments/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_binary.py || true
run_experiment "E1_5_step1_age" experiments/rq1_bidirectional/ramd/E1_5_ramd_oof_step1_age_4class.py || true
run_experiment "E1_4_E1_5_step2" experiments/rq1_bidirectional/ramd/E1_4_E1_5_ramd_step2.py || true

echo ""
echo "--- Phase 5: RQ2 D1 — Teacher signals ---"
run_experiment "E2_1_reverse_kl" experiments/rq2_d1_teacher_signals/response_based/E2_1_reverse_kl_seeds.py || true
run_experiment "E2_2_gender_llm4es" experiments/rq2_d1_teacher_signals/feature_based/E2_2_gender_llm4es.py || true
run_experiment "E2_2_rosbank_llm4es" experiments/rq2_d1_teacher_signals/feature_based/E2_2_rosbank_llm4es.py || true
run_experiment "E2_2_age_llm4es_v2" experiments/rq2_d1_teacher_signals/feature_based/E2_2_age_llm4es_v2.py || true
run_experiment "E2_4_combined" experiments/rq2_d1_teacher_signals/combined/E2_4_combined_kd.py || true

echo ""
echo "--- Phase 6: RQ3 — LLM size effect ---"
run_experiment "E5_x_qwen25_3b_extract" \
    experiments/rq3_llm_size_effect/E5_x_extract_llm_embeddings.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --teacher "qwen25_3b" \
    --datasets gender rosbank age || true

echo ""
echo "--- Aggregating results ---"
python scripts/aggregate_results.py || true

echo ""
echo "=================================================================="
echo "Overnight run finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================================="
