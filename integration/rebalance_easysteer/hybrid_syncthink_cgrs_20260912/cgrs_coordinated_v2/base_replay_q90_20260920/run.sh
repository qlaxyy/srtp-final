set -euo pipefail
export PATH=/root/autodl-tmp/venvs/easysteer-vllm026/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
Q90_HOME=/root/autodl-tmp/projects/base_replay_q90_20260920
Q90_OUT=/root/autodl-tmp/results/easysteer/base_replay_q90_20260920_run1
Q90_RUNTIME=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
Q90_PY=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
mkdir "$Q90_OUT"
date -Is > "$Q90_OUT/started.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop-ms=1000 > "$Q90_OUT/gpu.csv" &
Q90_MONITOR=$!
trap 'kill "$Q90_MONITOR" 2>/dev/null || true' EXIT
for Q90_CASE in R L27; do
  "$Q90_PY" "$Q90_HOME/evaluate.py" --case "$Q90_CASE" --fit-dir "$Q90_HOME/${Q90_CASE}_fit" --runtime-root "$Q90_RUNTIME" --output "$Q90_OUT/${Q90_CASE}_math500"
done
for Q90_CASE in R L27; do
  /root/autodl-tmp/venvs/rebalance/bin/python "$Q90_HOME/grade.py" --run "$Q90_OUT/${Q90_CASE}_math500" --runtime-root "$Q90_RUNTIME"
done
date -Is > "$Q90_OUT/finished.txt"
echo Q90_BATCH_COMPLETE
