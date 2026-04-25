#!/usr/bin/env bash
# Sequential runner for RQ3 size ladder on Gender dataset.
#
# Order:
#   D1 (LATTE):  E5.0 (0.5B) → E5.1 (1.5B) → E5.2-Instruct (3B) → E5.3 (7B)
#   D2 (kNN):    E6.0 (0.5B) → E6.1 (1.5B) → E6.1.5 (3B)
#
# E6.2 (7B) уже есть в predecessor experiments — не повторяем.
#
# Logs: results/<output_dir>/run.log
# Failure: stops the whole pipeline (set -e). Re-running skips cached embeddings.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_BASE="${REPO_ROOT}/results"
mkdir -p "$LOG_BASE"

run_one () {
    local label="$1"
    local script="$2"
    local log_dir="$3"

    mkdir -p "$log_dir"
    local log_file="${log_dir}/run.log"

    echo ""
    echo "================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $label"
    echo "  script: $script"
    echo "  log:    $log_file"
    echo "================================================"

    if python -u "$script" 2>&1 | tee "$log_file"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $label"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $label  (see $log_file)"
        exit 1
    fi
}

# ---- D1: LATTE ladder (extract embeddings + train LATTE) ----
run_one "E5.0  Qwen2.5-0.5B  D1 LATTE" \
    "experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_0_gender_latte_qwen0_5b.py" \
    "${LOG_BASE}/gender_latte_qwen25_0_5b_instruct"

run_one "E5.1  Qwen2.5-1.5B  D1 LATTE" \
    "experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_1_gender_latte_qwen1_5b.py" \
    "${LOG_BASE}/gender_latte_qwen25_1_5b_instruct"

run_one "E5.2  Qwen2.5-3B-Instruct  D1 LATTE" \
    "experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_2_gender_latte_qwen3b_instruct.py" \
    "${LOG_BASE}/gender_latte_qwen25_3b_instruct"

run_one "E5.3  Qwen2.5-7B  D1 LATTE" \
    "experiments/rq3_llm_size_effect/d1_teacher_size_for_latte/E5_3_gender_latte_qwen7b.py" \
    "${LOG_BASE}/gender_latte_qwen25_7b_instruct"

# ---- D2: kNN-CoT ladder (inference only) ----
run_one "E6.0   Qwen2.5-0.5B  D2 kNN-CoT" \
    "experiments/rq3_llm_size_effect/d2_size_for_enrichment/qwen_25_0_5b/E6_0_gender_qwen0_5b.py" \
    "${LOG_BASE}/gender_d2_qwen25_0_5b"

run_one "E6.1   Qwen2.5-1.5B  D2 kNN-CoT" \
    "experiments/rq3_llm_size_effect/d2_size_for_enrichment/qwen_25_1_5b/E6_1_gender_qwen1_5b.py" \
    "${LOG_BASE}/gender_d2_qwen25_1_5b"

run_one "E6.1.5 Qwen2.5-3B    D2 kNN-CoT" \
    "experiments/rq3_llm_size_effect/d2_size_for_enrichment/qwen_25_3b/E6_1_5_gender_qwen3b.py" \
    "${LOG_BASE}/gender_d2_qwen25_3b"

echo ""
echo "================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] LADDER COMPLETE"
echo "  D1: 4 LATTE runs (Qwen2.5 0.5B / 1.5B / 3B-Instruct / 7B)"
echo "  D2: 3 kNN runs   (Qwen2.5 0.5B / 1.5B / 3B-Instruct)"
echo "================================================"
