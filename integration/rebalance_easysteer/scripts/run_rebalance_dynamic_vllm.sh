#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/venvs/easysteer-vllm026}"
LIMIT="${LIMIT:-20}"
OFFSET="${OFFSET:-0}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/results/easysteer/rebalance_dynamic_vllm_gsm8k${LIMIT}_offset${OFFSET}.json}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Python environment not found: ${ENV_DIR}" >&2
  exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
cd "${PROJECT_ROOT}"

"${ENV_DIR}/bin/python" integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --output "${OUTPUT}" \
  "$@"
