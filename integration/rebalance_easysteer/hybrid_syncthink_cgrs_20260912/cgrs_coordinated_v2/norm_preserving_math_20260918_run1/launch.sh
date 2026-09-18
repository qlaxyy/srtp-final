#!/bin/bash
set -euo pipefail
export PATH="/root/autodl-tmp/venvs/easysteer-vllm026/bin:$PATH"
project=/root/autodl-tmp/projects/norm_preserving_math_20260918_run1
runtime=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
output=/root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/norm_preserving_math_20260918_run1
echo '4fdff42a77b0e39d03efba2bfe4ff819e206d489b4e2a0f2f88d8c2d39a3dce0  /root/autodl-tmp/projects/norm_preserving_math_20260918_run1.tar.gz' | sha256sum -c -
test ! -e "$output"
mkdir "$project"
tar -xzf "$project.tar.gz" -C "$project"
cd "$project"
release="$project/norm_preserving_math_20260918_run1/release.json"
python check_norm_graph.py "$runtime/sources/EasySteer/vllm-steer/vllm/steer_vectors/graph_kernels.py" cuda > gpu_kernel_check.json
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/reference" --phase engineering --norm-reference > reference.log 2>&1
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/engineering" --phase engineering --norm-reference-result "$output/reference" > engineering.log 2>&1
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/full" --phase full --engineering-result "$output/engineering" > full.log 2>&1
/root/autodl-tmp/venvs/rebalance/bin/python grade_label_alignment.py --run "$output/full" --runtime-root "$runtime" > grade.log 2>&1
cp gpu_kernel_check.json "$output/"
tar -czf "$project.completed.tar.gz" -C /root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912 norm_preserving_math_20260918_run1
sha256sum "$project.completed.tar.gz"
