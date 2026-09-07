#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/venvs/easysteer-vllm026}"
WHEEL_DIR="${WHEEL_DIR:-/root/autodl-tmp/wheels}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
NETWORK_TURBO="${NETWORK_TURBO:-/etc/network_turbo}"

source /root/miniconda3/etc/profile.d/conda.sh
mkdir -p "${WHEEL_DIR}" "${PIP_CACHE_DIR}" \
  /root/autodl-tmp/{hf-cache,models,results/easysteer,venvs}
export PIP_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  conda create -p "${ENV_DIR}" python=3.12 -y
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel -i "${PYPI_INDEX}"

if ! command -v aria2c >/dev/null 2>&1; then
  apt-get update
  apt-get install -y aria2
fi

download_direct() {
  local output="$1"
  local url="$2"
  echo "Downloading from the official upstream CDN: ${output}"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --max-tries=20 \
      --retry-wait=5 \
      --connect-timeout=30 \
      --timeout=60 \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --file-allocation=none \
      --dir="${WHEEL_DIR}" \
      --out="${output}" \
      "${url}"
}

download_github() {
  local output="$1"
  local url="$2"
  echo "Downloading from GitHub with temporary AutoDL academic acceleration: ${output}"
  (
    if [[ -f "${NETWORK_TURBO}" ]]; then
      # AutoDL only documents this proxy for GitHub/Hugging Face.  A subshell
      # prevents the proxy variables from leaking into PyTorch or pip traffic.
      source "${NETWORK_TURBO}"
    fi
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --max-tries=20 \
      --retry-wait=5 \
      --connect-timeout=30 \
      --timeout=60 \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --file-allocation=none \
      --dir="${WHEEL_DIR}" \
      --out="${output}" \
      "${url}"
  )
}

verify_wheel() {
  local wheel="$1"
  "${PYTHON}" -m zipfile -t "${WHEEL_DIR}/${wheel}" >/dev/null
  echo "Wheel integrity OK: ${wheel}"
}

download_github vllm-0.26.0-cu129-x86_64.whl \
  'https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
verify_wheel vllm-0.26.0-cu129-x86_64.whl

# PyTorch's own CDN was fast on the verified AutoDL instance.  Do not route it
# through network_turbo and do not fall back to generic PyPI CUDA wheels.
download_direct torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torch-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download_direct torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torchvision-0.26.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download_direct torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torchaudio-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download_direct torchcodec-0.16.0+cu129-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_28_x86_64.whl \
  'https://download.pytorch.org/whl/cu129/torchcodec-0.16.0%2Bcu129-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
download_direct triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

verify_wheel torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
verify_wheel torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
verify_wheel torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
verify_wheel torchcodec-0.16.0+cu129-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_28_x86_64.whl
verify_wheel triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

"${PYTHON}" -m pip install \
  "${WHEEL_DIR}"/triton-3.6.0-*.whl \
  "${WHEEL_DIR}"/torch-2.11.0+cu129-*.whl \
  "${WHEEL_DIR}"/torchvision-0.26.0+cu129-*.whl \
  "${WHEEL_DIR}"/torchaudio-2.11.0+cu129-*.whl \
  "${WHEEL_DIR}"/torchcodec-0.16.0+cu129-*.whl \
  -i "${PYPI_INDEX}"

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu129
export VLLM_PRECOMPILED_WHEEL_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0
export VLLM_PRECOMPILED_WHEEL_LOCATION="${WHEEL_DIR}/vllm-0.26.0-cu129-x86_64.whl"
# vllm-steer is vendored inside this monorepo, so setuptools-scm cannot infer
# its former standalone Git version.  Keep the verified upstream version.
export VLLM_VERSION_OVERRIDE=0.1.dev18960+g6267ca0cf

"${PYTHON}" -m pip install --editable "${PROJECT_ROOT}/sources/EasySteer/vllm-steer" \
  --find-links "${WHEEL_DIR}" -i "${PYPI_INDEX}"
"${PYTHON}" -m pip install --editable "${PROJECT_ROOT}/sources/EasySteer" \
  gguf 'math-verify[antlr4_13_2]==0.9.0' ninja \
  -i "${PYPI_INDEX}"

"${PYTHON}" -m pip check
"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_environment.py"
