# SEAL 机制复现

## 范围

本项目只复现 EasySteer 仓库中与减少过度思考相关的 SEAL steering 机制，用来验证框架链路和配对评测。它不是 SEAL 论文的完整训练、向量构造和论文等价复现。

## 运行

脚本会在同一批 MATH-500 题目上依次运行 baseline 与 SEAL，并报告准确率、平均生成 token、截断数和时间。

```bash
cd /root/autodl-tmp/projects/srtp-final
SEAL_LIMIT=20 bash integration/rebalance_easysteer/scripts/run_seal_math500.sh
SEAL_LIMIT=100 bash integration/rebalance_easysteer/scripts/run_seal_math500.sh
```

默认数据路径为 `/root/autodl-tmp/datasets/math500/test.jsonl`。首次缺失时评测脚本会下载数据；如果网络较慢，应单独完成数据下载后再运行，不要把“下载无输出”误判为 GPU 推理卡死。

## 当前结果

100题：baseline 准确率72.0%、平均4125.3 token；SEAL准确率75.0%、平均3248.4 token。token 减少21.3%，准确率增加3.0个百分点，时间102.5→106.2秒。

该结果证明当前机制能改变生成长度和答案，但不能独立支撑论文级效果声明。正式研究主线仍是 ReBalance 适配与后续优化。
