# 开发入门

main已在PR #4合入B线研究代码。代码合入与候选晋级是不同事项；当前冻结参照仍为ReBalance＋L27，最新单侧端点候选只完成CPU准备。

## 获取代码与选择起点

```bash
git clone https://github.com/qlaxyy/srtp-final.git
cd srtp-final
git switch -c codex/your-topic
git rev-parse HEAD
git status --short
```

记录起点提交。一般开发从最新main建分支；复核冻结结果时，先与维护者核对对应标签、运行代码及资产哈希。不要切换其他人的工作目录、覆盖已存在的结果或移动冻结标签。

## 最小阅读路线

1. [项目进展](research/PROJECT_STATUS.md)：明确当前方法、基准和未验证内容。
2. [研究交接](research/00-研究交接.md)：先读顶部最新记录，避免重复失败实验。
3. [运行手册](research/03-ReBalance适配与运行.md)：查校准和评测入口；历史配置不是最新候选的运行命令，具体以候选协议为准。
4. [贡献说明](../CONTRIBUTING.md)：提交方案、数据登记和实验约束。

## 关键代码入口

| 任务 | 入口 |
|---|---|
| 基线步骤处理、选层与向量拟合 | [calibrate_auto.py](../integration/rebalance_easysteer/scripts/calibrate_auto.py) |
| vLLM动态控制 | [rebalance.py](../sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py) |
| 组合实验运行入口 | [run_label_alignment.py](../integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/run_label_alignment.py) |
| L27协调与状态接入 | [label_alignment_adapter.py](../integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/label_alignment_adapter.py) |
| 词表短语匹配 | [lexicon_automaton.py](../integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/lexicon_automaton.py) |
| 最新CPU诊断示例 | [endpoint_audit_20260920](../integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/endpoint_audit_20260920) |

上述为阅读入口，部分脚本依赖既有模型、缓存和绝对路径；先查协议和参数，不把它们当作通用开箱即用命令。修改共享控制器前，在PR中列出影响范围和默认关闭时的行为。

## 环境与资产

- 文档和代码阅读只需Git；CPU分析所需Python依赖取决于脚本。
- GPU路径使用项目修改的EasySteer/vLLM，不能直接安装任意新版vLLM代替。已有环境先核验记录，不自动升级；确需新环境时参考[环境手册](research/01-AutoDL环境搭建.md)。
- 向维护者索取匹配的模型位置、向量/fit、数据题单、完整输出或隐藏状态及哈希。历史服务器端口不是公共服务入口。
- `.codex_work`中的大缓存不随仓库分发。仓库中的小型向量和摘要不代表完整实验依赖已齐备。

## 第一个贡献怎么做

先选择一个明确问题：复核某个指标、补一个CPU可视化、解释一个已知失败，或实现默认关闭的单因素候选。在Issue/PR中写清动机、文件、数据、验证与成本；先完成CPU工作，再协调GPU批次。

PR至少说明：基准提交、改变的因素、比较对象、已执行检查、未验证部分；实验PR另附模型/seed/上限/题数/配置/资产与输出哈希、正确率、思考与总token、触顶和逐题变化。失败结果同样保留。维护者审阅后合并，服务器部署另行协调。
