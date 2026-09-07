# 思维链压缩复现仓库

本仓库集中管理 EasySteer、ReBalance 参考源码，以及两者结合后的评测与运行入口。目标是减少推理模型的过度思考，并同时比较准确率、生成 token 数和运行时间。

## 当前进度

- EasySteer/vLLM 推理与 steering 链路已打通；
- SEAL 的框架机制已完成小规模配对实验；
- ReBalance 静态版已接入 EasySteer；
- ReBalance 作者代码逻辑的动态版已接入，并实现逐请求状态隔离；
- 动态版保持了准确率，但速度仍需优化。

实验结论、版本区别和后续任务统一见 [研究交接](docs/research/00-研究交接.md)。

## 目录

```text
sources/EasySteer/                 实际运行的 EasySteer，含修改后的 vLLM
sources/ReBalance/                 作者 ReBalance 源码快照，用于核对
integration/rebalance_easysteer/   配对评测和统一运行脚本
scripts/                           AutoDL 安装、更新和环境验收
env/                               已验证版本
docs/research/                     四份中文研究文档
```

不再另外保存一份“未修改 EasySteer”。上游代码与本项目修改的区别由 Git 历史记录；必须修改推理核心的动态控制代码位于 `sources/EasySteer/vllm-steer`。

## AutoDL 快速开始

首次使用：

```bash
cd /root/autodl-tmp/projects
git clone https://github.com/qlaxyy/srtp-final.git
cd srtp-final
bash scripts/setup_autodl.sh
```

已有环境只更新代码：

```bash
bash /root/autodl-tmp/projects/srtp-final/scripts/update_server.sh
```

20题低成本验收：

```bash
cd /root/autodl-tmp/projects/srtp-final
LIMIT=20 bash integration/rebalance_easysteer/scripts/run_rebalance_static_vllm.sh
LIMIT=20 bash integration/rebalance_easysteer/scripts/run_rebalance_dynamic_vllm.sh
```

模型、向量、完整数据、Python 环境和逐题结果均保存在 `/root/autodl-tmp`，不提交 Git。默认外部路径及覆盖方法见 [ReBalance 运行说明](docs/research/03-ReBalance适配与运行.md)。

## 协作方式

首次基线建立后，所有修改统一采用：本地功能分支 → 推送 GitHub → PR 审查合并 → 服务器 `git pull --ff-only`。不再发送 ZIP，也不再手工覆盖服务器源码。
