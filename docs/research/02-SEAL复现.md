# SEAL 机制复现

用途：已完成的SEAL机制验证历史说明。下列命令仅供定位或有明确需求时复跑，不是接手当前ReBalance基线的前置步骤；最新状态见[00-研究交接.md](00-研究交接.md)。

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

## 历史结果（暂不纳入方法对比）

2026-09-10用户决定：SEAL暂不放入当前方法对比表，以后另行补测。现有记录保留，本次不启动补测。

1.5B的MATH-500前100题：无干预准确率72.00%、平均4125.300 token；SEAL准确率75.00%、平均3248.410 token。准确率+3.00pp、全部生成token−21.26%，触顶36→24题，时间102.5→106.2秒。

这轮使用8192生成上限、temperature=0、seed=42和math-verify判分，未单独记录思考token；它是EasySteer示例方向的机制适配，不是SEAL论文全流程复现。与当前ReBalance的全量、16000上限、采样生成和作者判分协议不同，不能直接排名。SEAL尚无全量500题、7B或GSM8K结果；未来需要对比时再统一协议补测，不在接手时自动执行。

来源：本地`.codex_work/server-audit/seal_math500_n100_offset0.json`（服务器`/root/autodl-tmp/results/easysteer/`下同名文件），SHA256为`6cc0a0add0568f6707c31a012202eac116421036448994de1eb9201529ab2a83`。9月10日已按100份逐题记录复算正确数、token和触顶数；历史产物保持不变。
