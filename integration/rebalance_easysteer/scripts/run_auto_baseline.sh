#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"
MODEL="${MODEL:?Set the local model directory}"
SOURCE="${CALIBRATION_SOURCE:?Set the directory for this model calibration generations}"
OUT="${OUTPUT:?Set a separate calibration and result directory}"
VLLM=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
LEGACY=/root/autodl-tmp/venvs/rebalance/bin/python
export PATH="$(dirname "$VLLM"):$PATH"
export PYTHONNOUSERSITE=1
cd "$ROOT"
mkdir -p "$OUT"

if [[ "${1:-}" != --group ]]; then
    common=(--source "$SOURCE" --output "$OUT" --model "$MODEL")
    if [[ ! -f "$SOURCE/generation_summary.json" ]]; then
        resume=()
        [[ "${RESUME_CALIBRATION:-0}" != 1 ]] || resume=(--resume-generation)
        timeout --signal=TERM --kill-after=10 "${CALIBRATION_TIMEOUT_SECONDS:-1500}" "$VLLM" -u \
            integration/rebalance_easysteer/scripts/calibrate_auto.py generate "${common[@]}" "${resume[@]}" \
            > "$OUT/generate.log" 2>&1
    fi
    for stage in prepare collect select fit; do
        case "$stage" in
            prepare) marker=protocol.json ;;
            collect) marker=collection.json ;;
            select) marker=selected_layer.json ;;
            fit) marker=fit.json ;;
        esac
        if [[ ! -f "$OUT/$marker" ]]; then
            extra=()
            [[ -z "${FEATURE_CACHE:-}" ]] || extra=(--feature-cache "$FEATURE_CACHE")
            [[ -z "${FEATURE_DIR:-}" ]] || extra+=(--feature-dir "$FEATURE_DIR")
            timeout --signal=TERM --kill-after=10 1500 "$LEGACY" -u \
                integration/rebalance_easysteer/scripts/calibrate_auto.py "$stage" \
                "${common[@]}" "${extra[@]}" > "$OUT/${stage}.log" 2>&1
        fi
    done
    for dataset in ${BENCHMARKS:-math500 gsm8k}; do
        # Each arm has its own 1500s generation+grading budget inside Python.
        # A new model needs two arms; neither is allowed to borrow the other's time.
        # The outer cap additionally bounds startup and cleanup.
        timeout --signal=TERM --kill-after=10 3300 bash "$0" --group "$dataset"
    done
    echo "Automatic baseline completed $(date -Is)"
    exit 0
fi

dataset="$2"
case "$dataset" in
    math500)
        data="$ROOT/sources/ReBalance/Data/Math_Math500/test.jsonl"
        max_tokens="${EVAL_MAX_TOKENS_MATH500:-8192}"
        count=500; grade=math; baseline="${BASELINE_MATH500:-}" ;;
    gsm8k)
        data="$ROOT/sources/ReBalance/Data/Math_GSM8K/test.jsonl"
        max_tokens="${EVAL_MAX_TOKENS_GSM8K:-4096}"
        count=1319; grade=Math_GSM8K; baseline="${BASELINE_GSM8K:-}" ;;
    *) exit 2 ;;
esac
[[ ! -e "$OUT/${dataset}_eval.json" ]] || { echo "Result exists: $dataset"; exit 2; }
reuse=()
[[ -z "$baseline" ]] || reuse=(--baseline-result "$baseline")
[[ -n "$baseline" ]] || reuse+=(--dynamic-first)
[[ "$max_tokens" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid generation limit"; exit 2; }
# The evaluator checks every actual prompt fits; never silently truncate inputs.
context="${EVAL_MAX_MODEL_LEN:-$((max_tokens + 1024))}"
concurrency="${EVAL_MAX_NUM_SEQS:-32}"
prefill=()
if [[ "${EVAL_CHUNKED_PREFILL:-1}" == 1 ]]; then
    prefill=(--chunked-prefill --max-num-batched-tokens "${EVAL_MAX_BATCHED_TOKENS:-2048}")
fi
if [[ -f "$OUT/eval_runtime.json" ]]; then
    context=$("$LEGACY" -c 'import json,sys; from pathlib import Path; c=json.load(open(sys.argv[1])); assert Path(c["model"]).resolve()==Path(sys.argv[2]).resolve(); n=c["max_model_len"]; assert isinstance(n,int) and n>int(sys.argv[3]); print(n)' "$OUT/eval_runtime.json" "$MODEL" "$max_tokens")
    concurrency=$("$LEGACY" -c 'import json,sys; n=json.load(open(sys.argv[1])).get("max_num_seqs",256); assert isinstance(n,int) and n>0; print(n)' "$OUT/eval_runtime.json")
fi
artifacts="${EVAL_ARTIFACTS_DIR:-$OUT}"
echo "$dataset: count=$count, max_tokens=$max_tokens, context=$context, concurrency=$concurrency; started $(date -Is)"
"$VLLM" -u integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py \
    --model "$MODEL" --dataset "$data" --limit "$count" \
    --max-tokens "$max_tokens" --max-model-len "$context" \
    --max-num-seqs "$concurrency" \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.92}" \
    "${prefill[@]}" \
    --vector "$artifacts/auto_vector.pt" --calibration-fit "$artifacts/fit.json" \
    "${reuse[@]}" --output "$OUT/${dataset}_eval.json" \
    --group-timeout-seconds 1500 > "$OUT/${dataset}_eval.log" 2>&1
"$LEGACY" -u integration/rebalance_easysteer/scripts/regrade_saved_results.py \
    --input "$OUT/${dataset}_eval.json" --output "$OUT/${dataset}_author_grading.json" \
    --data-name "$grade" > "$OUT/${dataset}_grading.log" 2>&1
echo "$dataset completed $(date -Is)"
