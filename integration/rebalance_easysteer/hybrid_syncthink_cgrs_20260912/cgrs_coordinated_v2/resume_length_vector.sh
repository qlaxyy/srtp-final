#!/bin/bash
set -euo pipefail
export PATH="/root/autodl-tmp/venvs/easysteer-vllm026/bin:$PATH"
project=/root/autodl-tmp/projects/trajectory_length_vector_20260918_run2
runtime=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
output=/root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/trajectory_length_vector_20260918_run2
cd "$project"
release="$project/trajectory_length_vector_20260918_run2/release.json"
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/engineering_retry1" --phase engineering
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/full" --phase full --engineering-result "$output/engineering_retry1"
/root/autodl-tmp/venvs/rebalance/bin/python grade_label_alignment.py --run "$output/full" --runtime-root "$runtime"
tar -czf "$project.completed.tar.gz" -C /root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912 trajectory_length_vector_20260918_run2
sha256sum "$project.completed.tar.gz"
