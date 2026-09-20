set -euo pipefail
export PATH=/root/autodl-tmp/venvs/easysteer-vllm026/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CROSS_HOME=/root/autodl-tmp/projects/vector_curve_cross_20260920
CROSS_OUT=/root/autodl-tmp/results/easysteer/vector_curve_cross_20260920_run1
CROSS_RUNTIME=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
mkdir "$CROSS_OUT"
date -Is > "$CROSS_OUT/started.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop-ms=1000 > "$CROSS_OUT/gpu.csv" &
CROSS_MONITOR=$!
trap 'CROSS_STATUS=$?; kill "$CROSS_MONITOR" 2>/dev/null || true; printf "%s\n" "$CROSS_STATUS" > "$CROSS_OUT/exit_status.txt"' EXIT
for CROSS_ARM in old_vector_new_curve new_vector_old_curve; do
  /root/autodl-tmp/venvs/easysteer-vllm026/bin/python "$CROSS_HOME/evaluate.py" --case L27 --arm "$CROSS_ARM" --fit-dir "$CROSS_HOME/assets/$CROSS_ARM" --runtime-root "$CROSS_RUNTIME" --output "$CROSS_OUT/$CROSS_ARM"
  /root/autodl-tmp/venvs/rebalance/bin/python "$CROSS_HOME/grade.py" --run "$CROSS_OUT/$CROSS_ARM" --runtime-root "$CROSS_RUNTIME"
done
date -Is > "$CROSS_OUT/finished.txt"
echo CROSS_BATCH_COMPLETE
