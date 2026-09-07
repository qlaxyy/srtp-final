# ReBalance 适配与运行

## 两个版本

2026-09-07更新：控制曲线移植差异已修正，251,251点CPU网格与作者源码一致；修正后20题系数、性能及token等价诊断完成，简要结果见 [00-研究交接.md](00-研究交接.md)。旧200题数据仍是修正前结果；原差异证据见 [04-接手核查与最小实验.md](04-接手核查与最小实验.md)。

静态版直接使用作者公开方向向量，在 EasySteer 层18的生成换段边界施加固定系数 `-1`。它故意不包含置信度控制，用来先验证向量、层映射、触发位置和 vLLM 加速。

动态版为每个请求维护独立状态：从原始 logits 计算 token 最大概率，在步骤边界汇总置信度和相邻步骤方差，再调用修正后的作者控制函数。它使用作者 Qwen2 代码中的算术平均，而非论文几何平均。静态版还会命中 think 结束后的换段，动态版用 think 状态屏蔽这些注入。

作者函数适合做原始方法复现。只有在收集自己的独立校准数据后，才应拟合新函数；新函数必须单列为改进方法，不能继续称为作者动态版。

## 默认外部资产

```text
模型: /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B
数据: /root/autodl-tmp/ReBalance/Data/Math_GSM8K_200_seed42/test.jsonl
向量: /root/autodl-tmp/ReBalance/vectors/DeepSeek-R1-Distill-Qwen-1.5B/steer_vector_layer19_conf_mixed.pt
结果: /root/autodl-tmp/results/easysteer
```

作者的“block 19 前注入”对应 EasySteer decoder output 的层18。模型、数据和向量不提交 Git。需要换路径时，直接在运行命令末尾传 `--model`、`--dataset`、`--vector` 或 `--output`。

## 运行顺序

先做不加载模型的动态状态验收：

```bash
cd /root/autodl-tmp/projects/srtp-final
/root/autodl-tmp/venvs/easysteer-vllm026/bin/python \
  integration/rebalance_easysteer/scripts/verify_rebalance_dynamic.py
```

再做20题配对冒烟：

```bash
LIMIT=20 bash integration/rebalance_easysteer/scripts/run_rebalance_static_vllm.sh
LIMIT=20 bash integration/rebalance_easysteer/scripts/run_rebalance_dynamic_vllm.sh
```

下面保留历史200题运行方式，新的扩大测评须经用户确认，不能因小测试通过就自动启动：

```bash
LIMIT=200 bash integration/rebalance_easysteer/scripts/run_rebalance_static_vllm.sh
LIMIT=200 bash integration/rebalance_easysteer/scripts/run_rebalance_dynamic_vllm.sh
```

脚本直接使用环境内 Python，不要求先执行 `activate`。如环境位置变化，可临时设置 `ENV_DIR=/新路径`；项目位置变化可设置 `PROJECT_ROOT=/新路径`。

## 结果解释

200题历史静态结果为 token 减少32.7%、准确率下降3.0个百分点。当前动态适配版 token 减少16.0%、准确率总数持平，但生成调用时间由28.786秒升到44.989秒（原始记录已复核）。这只是该次子集观测，不能证明普遍无损、函数严格等价，或将全部16.203秒差值归因于置信度计算。环境、逐题汇总及截断情况见 [05-服务器现场验收.md](05-服务器现场验收.md)。

下一步先优化状态路径的同步，保持公式和20题输出不变。不要在测试集反复调函数；论文公式版及独立校准集拟合之后再决定。
