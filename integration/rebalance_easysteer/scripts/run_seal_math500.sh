#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${EASYSTEER_ENV_DIR:-/root/autodl-tmp/venvs/easysteer-vllm026}"
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/projects/srtp-final}"
LIMIT="${SEAL_LIMIT:-20}"
OFFSET="${SEAL_OFFSET:-0}"
OUTPUT="${SEAL_OUTPUT:-/root/autodl-tmp/results/easysteer/seal_math500_n${LIMIT}_offset${OFFSET}.json}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Missing EasySteer environment: $ENV_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIR/integration/rebalance_easysteer/eval/seal_math500_eval.py" ]]; then
    echo "Missing evaluation script under: $PROJECT_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR"

PYTHON="$ENV_DIR/bin/python"
export PATH="$ENV_DIR/bin:$PATH"

if ! "$PYTHON" -c "import math_verify" >/dev/null 2>&1; then
    "$PYTHON" -m pip install --no-cache-dir 'math-verify[antlr4_13_2]==0.9.0'
fi
if ! command -v ninja >/dev/null 2>&1; then
    "$PYTHON" -m pip install --no-cache-dir ninja
fi

"$PYTHON" -m pip check
"$PYTHON" integration/rebalance_easysteer/eval/seal_math500_eval.py \
    --limit "$LIMIT" \
    --offset "$OFFSET" \
    --output "$OUTPUT" \
    "$@"
