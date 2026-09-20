set -euo pipefail
export PATH=/root/autodl-tmp/venvs/easysteer-vllm026/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NATIVE_HOME=/root/autodl-tmp/projects/native_q90_eval_20260920
NATIVE_OUT=/root/autodl-tmp/results/easysteer/native_q90_eval_20260920_run1
NATIVE_RUNTIME=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
NATIVE_PY=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
mkdir "$NATIVE_OUT"
date -Is > "$NATIVE_OUT/started.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop-ms=1000 > "$NATIVE_OUT/gpu.csv" &
NATIVE_MONITOR=$!
trap 'NATIVE_STATUS=$?; kill "$NATIVE_MONITOR" 2>/dev/null || true; printf "%s\n" "$NATIVE_STATUS" > "$NATIVE_OUT/exit_status.txt"' EXIT
for NATIVE_CASE in R L27; do
  for NATIVE_QUANTILE in q75 q90; do
    "$NATIVE_PY" "$NATIVE_HOME/evaluate.py" --case "$NATIVE_CASE" --quantile "$NATIVE_QUANTILE" --fit-dir "$NATIVE_HOME/assets/$NATIVE_CASE/$NATIVE_QUANTILE" --runtime-root "$NATIVE_RUNTIME" --output "$NATIVE_OUT/${NATIVE_CASE}_${NATIVE_QUANTILE}_math500"
    /root/autodl-tmp/venvs/rebalance/bin/python "$NATIVE_HOME/grade.py" --run "$NATIVE_OUT/${NATIVE_CASE}_${NATIVE_QUANTILE}_math500" --runtime-root "$NATIVE_RUNTIME"
  done
done
date -Is > "$NATIVE_OUT/finished.txt"
echo NATIVE_Q90_BATCH_COMPLETE
