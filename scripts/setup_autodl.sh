#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/srtp-final}"
ENV_DIR="${ENV_DIR:-/root/autodl-tmp/venvs/easysteer-vllm026}"
WHEEL_DIR="${WHEEL_DIR:-/root/autodl-tmp/wheels}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source /root/miniconda3/etc/profile.d/conda.sh
mkdir -p "${WHEEL_DIR}" /root/autodl-tmp/{hf-cache,models,results/easysteer,venvs}

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  conda create -p "${ENV_DIR}" python=3.12 -y
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel -i "${PYPI_INDEX}"

if ! command -v aria2c >/dev/null 2>&1; then
  apt-get update
  apt-get install -y aria2
fi

download() {
  local output="$1"
  local url="$2"
  [[ -s "${WHEEL_DIR}/${output}" ]] || aria2c -c -x 16 -s 16 -k 1M -d "${WHEEL_DIR}" -o "${output}" "${url}"
}

download vllm-0.26.0-cu129-x86_64.whl \
  'https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
download torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torch-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torchvision-0.26.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/cu129/torchaudio-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl'
download torchcodec-0.16.0+cu129-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_28_x86_64.whl \
  'https://download.pytorch.org/whl/cu129/torchcodec-0.16.0%2Bcu129-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
download triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  'https://download-r2.pytorch.org/whl/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

"${PYTHON}" -m pip install --no-cache-dir \
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

"${PYTHON}" -m pip install --editable "${PROJECT_ROOT}/sources/EasySteer/vllm-steer" \
  --find-links "${WHEEL_DIR}" -i "${PYPI_INDEX}" --no-cache-dir
"${PYTHON}" -m pip install --editable "${PROJECT_ROOT}/sources/EasySteer" \
  gguf 'math-verify[antlr4_13_2]==0.9.0' ninja \
  -i "${PYPI_INDEX}" --no-cache-dir

"${PYTHON}" -m pip check
"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_environment.py"
