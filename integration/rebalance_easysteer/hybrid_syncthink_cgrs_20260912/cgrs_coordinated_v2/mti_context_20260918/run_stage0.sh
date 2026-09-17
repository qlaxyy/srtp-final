#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
runtime=/root/autodl-tmp/venvs/rebalance/bin/python
if [[ ! -x "$runtime" ]]; then
  echo "Pinned existing HF runtime missing; stop without installing."
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU already has a compute process; stop."
  exit 1
fi
exec timeout --signal=TERM --kill-after=10s 620s "$runtime" -u \
  mti_branch_diagnostic.py --plan stage0_plan.json \
  --output-root /root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/mti_context_20260918 \
  --gpu-authorized
