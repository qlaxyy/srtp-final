#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/EasySteer}"
VLLM_ROOT="${PROJECT_ROOT}/vllm-steer"
OVERLAY_ROOT="${BUNDLE_ROOT}/.codex_work/vllm-steer-audit2"
BACKUP_ROOT="${PROJECT_ROOT}/.rebalance-backup-$(date +%Y%m%d-%H%M%S)"

FILES=(
  vllm/forward_context.py
  vllm/steer_vectors/algorithms/__init__.py
  vllm/steer_vectors/algorithms/rebalance.py
  vllm/steer_vectors/api.py
  vllm/steer_vectors/payloads.py
  vllm/steer_vectors/rebalance.py
  vllm/steer_vectors/request.py
  vllm/v1/worker/gpu/model_runner.py
  vllm/v1/worker/gpu/steer_vector_utils.py
  tests/steer_vectors/test_rebalance.py
)

if [[ ! -d "${VLLM_ROOT}/vllm/steer_vectors" ]]; then
  echo "EasySteer vLLM submodule not found: ${VLLM_ROOT}" >&2
  exit 1
fi

for relative in "${FILES[@]}"; do
  source_file="${OVERLAY_ROOT}/${relative}"
  target_file="${VLLM_ROOT}/${relative}"
  if [[ ! -f "${source_file}" ]]; then
    echo "Bundle is incomplete: ${source_file}" >&2
    exit 1
  fi
  if [[ -f "${target_file}" ]]; then
    mkdir -p "${BACKUP_ROOT}/$(dirname "${relative}")"
    cp -a "${target_file}" "${BACKUP_ROOT}/${relative}"
  fi
  mkdir -p "$(dirname "${target_file}")"
  cp -a "${source_file}" "${target_file}"
done

cp -a \
  "${BUNDLE_ROOT}/replications/rebalance/rebalance_dynamic_eval.py" \
  "${PROJECT_ROOT}/replications/rebalance/rebalance_dynamic_eval.py"
cp -a \
  "${BUNDLE_ROOT}/replications/rebalance/run_rebalance_dynamic_vllm.sh" \
  "${PROJECT_ROOT}/replications/rebalance/run_rebalance_dynamic_vllm.sh"
cp -a \
  "${BUNDLE_ROOT}/replications/rebalance/verify_rebalance_dynamic.py" \
  "${PROJECT_ROOT}/replications/rebalance/verify_rebalance_dynamic.py"
chmod +x "${PROJECT_ROOT}/replications/rebalance/run_rebalance_dynamic_vllm.sh"

echo "Installed ReBalance dynamic vLLM files."
echo "Backup of replaced vLLM files: ${BACKUP_ROOT}"
echo "Next: run the verification commands from the reproduction guide."
