# ReBalance 适配与运行

## 两个版本

静态版直接使用作者公开方向向量，在 EasySteer 层18的生成换段边界施加固定系数 `-1`。它故意不包含置信度控制，用来先验证向量、层映射、触发位置和 vLLM 加速。

动态版在此基础上为每个请求维护独立状态：从原始 logits 计算 token 最大概率，在步骤边界汇总置信度和相邻步骤方差，再调用作者公开动态函数调整系数。它实现的是作者 Qwen2 代码中的算术平均，而非论文公式中的几何平均。

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

验收后扩大到200题：

```bash
LIMIT=200 bash integration/rebalance_easysteer/scripts/run_rebalance_static_vllm.sh
LIMIT=200 bash integration/rebalance_easysteer/scripts/run_rebalance_dynamic_vllm.sh
```

脚本直接使用环境内 Python，不要求先执行 `activate`。如环境位置变化，可临时设置 `ENV_DIR=/新路径`；项目位置变化可设置 `PROJECT_ROOT=/新路径`。

## 结果解释

200题静态版把 token 减少32.7%，准确率下降3.0个百分点；这是较强压缩、轻微损失的折中点。动态版把 token 减少16.0%且准确率不变，但时间由28.9秒升到45.0秒，说明当前动态控制在质量上有效、工程上尚未加速。

下一步不应直接在测试集反复调函数。应先优化逐 token 的全词表置信度与状态更新开销，再划分独立校准集拟合参数，最终只在保留测试集上评测。
