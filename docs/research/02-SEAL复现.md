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

## 方法结果汇总

2026-09-10核对本地保存的SEAL逐题结果后，与已冻结ReBalance结果放在同一表中。按模型分组，列出单次回答准确率（%）和平均全部生成token；**SEAL历史100题与全量结果的协议不同，不能据此横向排名。** 这里都是本项目实测，不抄入论文表中的外部成绩。

| 模型 | 方法／实验范围 | MATH准确率↑ | MATH平均token↓ | GSM8K准确率↑ | GSM8K平均token↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 无干预（全量） | 84.40 | 4460.854 | 78.09 | 1173.166 |
| 1.5B | ReBalance自校准适配（全量） | 82.20 | 3554.202 | 79.00 | 846.356 |
| 1.5B | 无干预（SEAL历史100题） | 72.00 | 4125.300 | — | — |
| 1.5B | SEAL机制适配（历史100题） | 75.00 | 3248.410 | — | — |
| **DeepSeek-R1-Distill-Qwen-7B** | 无干预（全量） | 91.80 | 3842.052 | 90.30 | 1184.850 |
| 7B | ReBalance自校准适配（全量） | 92.00 | 3234.130 | 89.61 | 1005.394 |
| 7B | SEAL（未测） | — | — | — | — |

- 全量：MATH-500全部500题、GSM8K全部1319题；上限16000、temperature=0.7、top_p=0.95、seed=42，作者判分。ReBalance使用各模型独立500道训练校准题、自提向量和自动选层；思考token另见00。
- SEAL历史结果：仅1.5B、MATH-500前100题、上限8192、temperature=0、seed=42，math-verify判分；使用EasySteer示例方向的机制适配，未完成SEAL论文全流程复现。它自己的配对变化为准确率+3.00pp、全部生成token−21.26%；触顶36→24题，全部纳入统计，时间102.5→106.2秒。原文件没有单独记录思考token。
- `—`表示没有可填的已完成结果，不是0。SEAL尚无本项目全量500题成绩，也没有7B或GSM8K结果；AIME24未测，因此使用本项目已测的GSM8K作为第二组列。
- 若要做论文式同条件方法对比，还需在相同完整题集、提示、采样、16000上限和判分口径下补齐SEAL；仅重新判分旧100题不能消除样本和生成设置差异。本次只整理已有结果，没有启动新测评。

来源：SEAL本地`.codex_work/server-audit/seal_math500_n100_offset0.json`（服务器`/root/autodl-tmp/results/easysteer/`下同名文件），SHA256为`6cc0a0add0568f6707c31a012202eac116421036448994de1eb9201529ab2a83`；已按100份逐题记录复算正确数、token和触顶数。ReBalance来自[最终冻结清单](../../integration/rebalance_easysteer/configs/final_results_20260909.json)。
