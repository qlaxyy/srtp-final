# 思维链压缩复现仓库

**2026-09-20进度：当前冻结研究方案为ReBalance＋L27；自迭代与单侧端点更新仍在探索，未替换冻结版。**

- [项目进展与完整指标](docs/research/PROJECT_STATUS.md)：1.5B/7B结果、近期失败、当前候选和证据入口。
- [如何参与协作](CONTRIBUTING.md)：代码分支、数据登记、GPU批次和PR要求。
- [B线实验代码与记录](https://github.com/qlaxyy/srtp-final/tree/codex/hybrid-syncthink-cgrs-20260912)：已完成与未评测候选均有明确标记。main的运行代码仍是下述冻结基线，不代表所有研究代码已合并。

以下为原ReBalance基线说明；最新研究进展以上面的状态页为准。

本仓库管理EasySteer、ReBalance参考源码及两者结合后的自校准与评测流程。目标是缩短思维链，同时尽量减少准确率损失。

**当前基线已完成并冻结：DeepSeek-R1-Distill-Qwen-1.5B／7B，GSM8K全1319题、MATH-500全500题，四组完整无干预／动态配对。** 各模型使用自己的500道训练校准题提取向量、自动选层和确定参数，通过EasySteer在vLLM内动态注入。思考长度减少17.68%–35.04%，准确率变化−2.20至+0.91个百分点；不声称普遍无损或原论文完整复现。

- **接手必读：[00-研究交接](docs/research/00-研究交接.md)**：目标、完成结果、当前状态、版本、连接／资产位置、坑、后续计划和全部docs用途。
- 操作手册：[03-ReBalance适配与运行](docs/research/03-ReBalance适配与运行.md)；安装／迁移时才读[01-AutoDL环境搭建](docs/research/01-AutoDL环境搭建.md)。
- 冻结标签：`easysteer-rebalance-v2-final-20260909`；[最终指标与来源清单](integration/rebalance_easysteer/configs/final_results_20260909.json)。后续文档提交不改变标签。

接手不需要重新校准、跑20题冒烟或补测已完成数据。论文重建版、作者公开向量版保留为历史实验，当前基线是明确补全过的自校准公开代码适配。

## 目录

```text
sources/EasySteer/                 实际运行的EasySteer，含修改后的vLLM
sources/ReBalance/                 作者源码快照，供核对、离线处理与判分
integration/rebalance_easysteer/   校准、配对评测、冻结配置与运行入口
scripts/                           AutoDL安装、更新与环境验收
env/                               已验证版本
docs/research/                     当前交接／操作手册和标明日期的历史审计
docs/UPSTREAM.md                   上游来源与修改边界
papers/                            本地论文，不上传Git
```

模型、Python环境、隐藏特征和完整逐题结果在服务器数据盘及本地备份，不进Git。测试数据的仓库路径见00；Git只有源码和摘要，不能替代完整资产备份。动态控制进入了`sources/EasySteer/vllm-steer`推理核心，不能直接换成原版vLLM。

## 环境与代码更新

服务器仓库`/root/autodl-tmp/projects/srtp-final`。已有两个验收环境保持隔离，不重装；迁移优先克隆完整实例并带数据盘。只有完全空白实例才用`scripts/setup_autodl.sh`，该空白安装全流程尚未独立复验。

协作修改通过独立分支和PR提交，核对代码与资产版本后再整合。服务器更新由维护者协调，不自动拉取研究分支覆盖当前运行环境；不覆盖冻结结果或移动标签。查看结果、整理文档不启动GPU实验。
