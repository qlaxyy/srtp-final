# AutoDL 环境搭建与迁移

本项目采用“代码进 Git，重资产留数据盘”的方式。新服务器只需克隆仓库并运行统一脚本；已有数据盘无需重复下载模型和 wheel。

## 已验证平台

- RTX 4090 D 24 GB，Ubuntu 22.04；
- Python 3.12，PyTorch 2.11.0+cu129；
- 详细版本见 `env/versions.md`。

本流程只适用于 NVIDIA CUDA，不要安装 Ascend/CANN。此前的 `kernel_operator.h` 编译错误来自误走 Ascend 算子构建链路，与本项目无关。

## 新实例

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h / /root/autodl-tmp

mkdir -p /root/autodl-tmp/projects
cd /root/autodl-tmp/projects
git clone https://github.com/qlaxyy/srtp-final.git
cd srtp-final
bash scripts/setup_autodl.sh
```

私有仓库需要在服务器配置 GitHub 凭据。不要把 token 写入脚本或提交仓库。

安装脚本会：

1. 在数据盘创建独立 Python 3.12 环境；
2. 用 `aria2` 多连接、断点续传，把大 wheel 永久保存到 `/root/autodl-tmp/wheels`；
3. 显式安装同一套 cu129 的 torch、vision、audio、codec 和 Triton；
4. 以 editable 方式安装仓库内 EasySteer/vLLM；
5. 检查依赖、CUDA 扩展和 steering 导入。

仓库中的 vLLM 已由独立子仓库改为统一仓库内的固定源码，因此安装脚本会显式设置已验证版本 `0.1.dev18960+g6267ca0cf`，不再依赖内层 `.git` 推导版本。

## 下载源策略

不强行让所有文件来自同一个网站，而是给每类文件固定唯一来源，避免自动回退和 CUDA 混装：

| 内容 | 固定来源 | 学术加速 |
| --- | --- | --- |
| vLLM 预编译 wheel | GitHub Release | 仅下载这个文件时开启 |
| torch、torchvision、torchaudio、torchcodec、Triton | PyTorch 官方 CDN | 明确关闭代理 |
| pip、EasySteer 普通依赖 | 清华 PyPI 镜像 | 不开启 |
| GitHub/Hugging Face 后续资源 | 对应官方站点 | 访问慢时临时开启 |

AutoDL 官方说明内置学术加速仅面向 GitHub 和 Hugging Face，而且不承诺稳定，也建议不用时关闭。因此脚本把 GitHub 下载放进独立子进程；该文件完成后代理自动消失，不会污染 PyTorch CDN、清华镜像或模型推理。

所有大 wheel 使用固定文件名和 `aria2 --continue`。脚本即使看到已有文件也会重新核对下载状态：完整文件立即通过，半成品从 `.aria2` 进度继续，不再因为“文件非空”错误跳过。每个 wheel 下载后还会进行 ZIP 完整性检查；普通依赖缓存则保存在 `/root/autodl-tmp/pip-cache`。网络失败时进度不会丢失，重新执行同一脚本即可续传，不要换 URL 或手工删除临时文件。

第一次从零安装仍需下载约数 GB。后续更换实例但保留数据盘时，大 wheel 和模型会直接复用。

## 已有服务器更新代码

```bash
bash /root/autodl-tmp/projects/srtp-final/scripts/update_server.sh
```

脚本先显示本地改动，再执行 fast-forward 拉取。若服务器出现未提交修改，不要强制覆盖；先提交、备份或移回本地分支处理。

## 为什么之前反复出错

中断下载不是主要根因。真正的问题是通用 PyPI 混入 CUDA 13 的 `torchvision`、`torchaudio` 或 `torchcodec`，而 torch/vLLM 使用 cu129，因而出现 `torchvision::nms`、`libcudart.so.13`、`libnvrtc.so.13` 错误；EasySteer 还漏声明了 `gguf`。统一脚本已经固定这些版本。

不要通过补装 CUDA 13 动态库修补，也不要单独升级 torch 二进制组。出现问题先运行：

```bash
/root/autodl-tmp/venvs/easysteer-vllm026/bin/python -m pip check
/root/autodl-tmp/venvs/easysteer-vllm026/bin/python scripts/verify_environment.py
```

## 迁移边界

Git 管理：源码、脚本、配置、测试、中文文档和小型结果摘要。

数据盘管理：`venvs/`、`wheels/`、`models/`、`hf-cache/`、完整数据集及逐题结果。实例克隆或挂载旧数据盘后，只需拉取新代码；依赖版本未变时无需重装环境。

环境安装脚本已按本次成功命令合并，但尚未在完全空白的新实例做一次 clean-room 全流程复验；首次队友复现时应保留完整日志，并据此更新脚本。
