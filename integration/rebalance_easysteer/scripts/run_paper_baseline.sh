#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/projects/srtp-final
CAL=/root/autodl-tmp/results/easysteer/paper_calibration_500_20260908
OUT="${OUTPUT:-$CAL/formal_v1}"
VLLM=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
LEGACY=/root/autodl-tmp/venvs/rebalance/bin/python
export PATH="$(dirname "$VLLM"):$PATH"
export PYTHONNOUSERSITE=1
cd "$ROOT"
mkdir -p "$OUT"

if [[ "${1:-}" != --group ]]; then
    cp -n integration/rebalance_easysteer/configs/paper_reconstruction_v1.json "$OUT/fit.json"
    for dataset in math500 gsm8k; do
        # One complete generation + both graders must fit the user's 25 minutes.
        timeout --signal=TERM --kill-after=10 1500 bash "$0" --group "$dataset"
    done
    echo "Paper baseline completed $(date -Is)"
    exit 0
fi

dataset="$2"
case "$dataset" in
    math500)
        data="$ROOT/sources/ReBalance/Data/Math_Math500/test.jsonl"
        count=500; grade=math
        baseline=/root/autodl-tmp/results/easysteer/rebalance_math500_full_16000_20260907.json ;;
    gsm8k)
        data="$ROOT/sources/ReBalance/Data/Math_GSM8K/test.jsonl"
        count=1319; grade=Math_GSM8K
        baseline=/root/autodl-tmp/results/easysteer/rebalance_gsm8k_full_16000_20260908.json ;;
    *) exit 2 ;;
esac
# Never implicitly overwrite or repeat an interrupted generation.
[[ ! -e "$OUT/${dataset}_eval.json" ]] || { echo "Result exists: $dataset"; exit 2; }
echo "$dataset started $(date -Is)"
"$VLLM" -u integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py \
    --dataset "$data" --limit "$count" --max-tokens 16000 --max-model-len 32768 \
    --vector "$CAL/paper_unit_vector.pt" --paper-fit "$OUT/fit.json" \
    --baseline-result "$baseline" --output "$OUT/${dataset}_eval.json" \
    --group-timeout-seconds 1500 > "$OUT/${dataset}_eval.log" 2>&1
"$LEGACY" -u integration/rebalance_easysteer/scripts/regrade_saved_results.py \
    --input "$OUT/${dataset}_eval.json" --output "$OUT/${dataset}_author_grading.json" \
    --data-name "$grade" > "$OUT/${dataset}_grading.log" 2>&1
echo "$dataset completed $(date -Is)"
