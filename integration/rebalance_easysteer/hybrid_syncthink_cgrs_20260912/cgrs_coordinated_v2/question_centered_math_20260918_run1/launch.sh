#!/bin/bash
set -euo pipefail
export PATH="/root/autodl-tmp/venvs/easysteer-vllm026/bin:$PATH"
project=/root/autodl-tmp/projects/question_centered_math_20260918_run1
runtime=/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917
output=/root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/question_centered_math_20260918_run1
echo '7022e9a21f695fd76eb8dfdbbe9cd5821fdd8d9b1041374ea2c43db3a024949e  /root/autodl-tmp/projects/question_centered_math_20260918_run1.tar.gz' | sha256sum -c -
test ! -e "$output"
mkdir "$project"
tar -xzf "$project.tar.gz" -C "$project"
cd "$project"
release="$project/question_centered_math_20260918_run1/release.json"
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/engineering" --phase engineering > engineering.log 2>&1
python run_label_alignment.py --release "$release" --runtime-root "$runtime" --output "$output/full" --phase full --engineering-result "$output/engineering" > full.log 2>&1
/root/autodl-tmp/venvs/rebalance/bin/python grade_label_alignment.py --run "$output/full" --runtime-root "$runtime" > grade.log 2>&1
tar -czf "$project.completed.tar.gz" -C /root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912 question_centered_math_20260918_run1
sha256sum "$project.completed.tar.gz"
