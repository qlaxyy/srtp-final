#!/usr/bin/env bash
# Own 500-question train calibration and frozen full-dataset evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUTPUT:?Set OUTPUT to a new calibration directory}"
VLLM=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
LEGACY=/root/autodl-tmp/venvs/rebalance/bin/python
export PATH="$(dirname "$VLLM"):$PATH"
export PYTHONNOUSERSITE=1
cd "$ROOT"
mkdir -p "$OUT"

run_budgeted() {
    local remaining=1500
    if [[ -n "${DEADLINE_EPOCH:-}" ]]; then
        remaining=$(( DEADLINE_EPOCH - $(date +%s) ))
        (( remaining > 0 )) || { echo 'Session budget exhausted'; exit 124; }
        (( remaining <= 1500 )) || remaining=1500
    fi
    timeout --signal=TERM --kill-after=10 "$remaining" "$@"
}

if [[ -n "${GENERATE_PID:-}" ]]; then
    while kill -0 "$GENERATE_PID" 2>/dev/null; do
        if [[ -n "${DEADLINE_EPOCH:-}" ]] && (( $(date +%s) >= DEADLINE_EPOCH )); then
            echo 'Session budget exhausted waiting for generation'; exit 124
        fi
        sleep 5
    done
    [[ -f "$OUT/generation_summary.json" ]] || { echo 'Generation failed'; exit 1; }
fi
if [[ ! -f "$OUT/generation_summary.json" ]]; then
    run_budgeted "$VLLM" -u integration/rebalance_easysteer/scripts/calibrate_own_vector.py \
        generate --output "$OUT" > "$OUT/generate.log" 2>&1
fi
if [[ ! -f "$OUT/extraction_summary.json" ]]; then
    run_budgeted "$LEGACY" -u integration/rebalance_easysteer/scripts/calibrate_own_vector.py \
        extract --output "$OUT" > "$OUT/extract.log" 2>&1
fi
if [[ ! -f "$OUT/fit.json" ]]; then
    run_budgeted "$LEGACY" -u integration/rebalance_easysteer/scripts/calibrate_own_vector.py \
        fit --output "$OUT" > "$OUT/fit.log" 2>&1
fi
for dataset in ${BENCHMARKS:-math500 gsm8k}; do
    case "$dataset" in
        math500)
            data="$ROOT/sources/ReBalance/Data/Math_Math500/test.jsonl"
            count=500; grade_name=math
            baseline=/root/autodl-tmp/results/easysteer/rebalance_math500_full_16000_20260907.json ;;
        gsm8k)
            data="$ROOT/sources/ReBalance/Data/Math_GSM8K/test.jsonl"
            count=1319; grade_name=Math_GSM8K
            baseline=/root/autodl-tmp/results/easysteer/rebalance_gsm8k_full_16000_20260908.json ;;
        *) echo "Unknown benchmark: $dataset"; exit 2 ;;
    esac
    if [[ ! -f "$OUT/${dataset}_eval.json" ]]; then
        run_budgeted "$VLLM" -u integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py \
            --dataset "$data" --limit "$count" --max-tokens 16000 --max-model-len 32768 \
            --vector "$OUT/own_vector_layer19.pt" --calibration-fit "$OUT/fit.json" \
            --baseline-result "$baseline" --output "$OUT/${dataset}_eval.json" \
            --group-timeout-seconds 1500 > "$OUT/${dataset}_eval.log" 2>&1
    fi
    if [[ ! -f "$OUT/${dataset}_author_grading.json" ]]; then
        run_budgeted "$LEGACY" -u integration/rebalance_easysteer/scripts/regrade_saved_results.py \
            --input "$OUT/${dataset}_eval.json" --output "$OUT/${dataset}_author_grading.json" \
            --data-name "$grade_name" > "$OUT/${dataset}_grading.log" 2>&1
    fi
    echo "$dataset completed $(date -Is)"
done
echo "Pipeline completed $(date -Is)"
