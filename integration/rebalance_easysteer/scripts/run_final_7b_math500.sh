#!/usr/bin/env bash
# Frozen-method MATH-500 completion; existing GSM8K results are not rerun.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"
export MODEL=/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
export EVAL_ARTIFACTS_DIR=/root/autodl-tmp/results/easysteer/auto_code_v2_qwen7b_20260908
export CALIBRATION_SOURCE="$EVAL_ARTIFACTS_DIR/calibration"
export OUTPUT="${OUTPUT:?Set a NEW directory for this final paired evaluation}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT"; exit 2; }
export PYTHONNOUSERSITE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export EVAL_MAX_TOKENS_MATH500=16000
export EVAL_MAX_MODEL_LEN=17408
export EVAL_MAX_NUM_SEQS=64
export EVAL_CHUNKED_PREFILL=1
export EVAL_MAX_BATCHED_TOKENS=2048
export EVAL_GPU_MEMORY_UTILIZATION=0.92
unset BASELINE_MATH500
cd "$PROJECT_ROOT"
# Both arms freshly generated, dynamic first; 1500s per arm including graders.
# Sampling defaults in the frozen evaluator: temperature .7, top_p .95, seed 42.
timeout --signal=TERM --kill-after=10 3300 bash \
    integration/rebalance_easysteer/scripts/run_auto_baseline.sh --group math500
