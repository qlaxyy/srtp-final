#!/bin/bash
set -euo pipefail
export PATH="/root/autodl-tmp/venvs/easysteer-vllm026/bin:$PATH"
project=/root/autodl-tmp/projects/length_sign_ablation_20260918_run1
runtime=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
output=/root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/length_sign_ablation_20260918_run1
mkdir "$project"
tar -xzf "$project.tar.gz" -C "$project"
cd "$project"
release="$project/length_sign_ablation_20260918_run1/release.json"
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/engineering" --phase engineering
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/full" --phase full --engineering-result "$output/engineering"
/root/autodl-tmp/venvs/rebalance/bin/python grade_label_alignment.py --run "$output/full" --runtime-root "$runtime"
tar -czf "$project.completed.tar.gz" -C /root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912 length_sign_ablation_20260918_run1
sha256sum "$project.completed.tar.gz"
