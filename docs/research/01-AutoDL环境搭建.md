# AutoDL 环境搭建与迁移

用途：安装、迁移或排查环境故障时查阅。**当前两套环境已经验收，接手不要重新安装。** 最新任务、连接、模型与结果位置见[00-研究交接.md](00-研究交接.md)，当前校准／评测入口见[03-ReBalance适配与运行.md](03-ReBalance适配与运行.md)。

本项目采用“代码进 Git，重资产留数据盘”的方式。自己的服务器迁移优先克隆完整实例；只有完全空白的新服务器才运行环境安装脚本。

## 已验证平台

- RTX 4090 D 24 GB，Ubuntu 22.04；
- Python 3.12，PyTorch 2.11.0+cu129；
- 详细版本见 `env/versions.md`。

本流程只适用于 NVIDIA CUDA，不要安装 Ascend/CANN。此前的 `kernel_operator.h` 编译错误来自误走 Ascend 算子构建链路，与本项目无关。

## 场景一：自己的实例迁移（默认工作流）

如果当前可租用实例都被占用，就在 AutoDL 控制台：

1. 选择最新一次使用的实例；
2. 点击“克隆实例”；
3. 勾选同时克隆数据盘；
4. 启动克隆后的新实例，直接继续实验。

完整克隆会复制系统盘上的 Miniconda、SSH 配置和系统工具，也会复制数据盘上的仓库、模型、Python 环境、wheel、数据和结果。因此克隆后不需要运行 `setup_autodl.sh`。如果克隆源在克隆前已经是 GitHub 最新代码，也不需要执行任何命令。

下面只是可选的更新和检查：

```bash
cd /root/autodl-tmp/projects/srtp-final
git pull --ff-only                         # 仅当 GitHub 有更新

/root/autodl-tmp/venvs/easysteer-vllm026/bin/python \
  scripts/verify_environment.py            # 仅当需要检查环境
```

如果存在一台以前创建、目前可以直接启动的实例，可以先启动它并从 GitHub 拉取代码；但 Git 只能更新源码，不能补齐模型、环境、向量和数据。缺少这些资产时，仍应克隆最新完整实例。

## 场景二：队友的完全空白实例

队友没有现成系统盘和数据盘可以克隆时，才执行：

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
| DeepSeek模型（国内服务器） | ModelScope，保存到 `/root/autodl-tmp/models/` | 不开启；已有1.5B／7B不重复下载 |
| GitHub/Hugging Face 后续资源 | 对应官方站点 | 访问慢时临时开启 |

AutoDL 官方说明内置学术加速仅面向 GitHub 和 Hugging Face，而且不承诺稳定，也建议不用时关闭。因此脚本把 GitHub 下载放进独立子进程；该文件完成后代理自动消失，不会污染 PyTorch CDN、清华镜像或模型推理。

所有大 wheel 使用固定文件名和 `aria2 --continue`。脚本即使看到已有文件也会重新核对下载状态：完整文件立即通过，半成品从 `.aria2` 进度继续，不再因为“文件非空”错误跳过。每个 wheel 下载后还会进行 ZIP 完整性检查；普通依赖缓存则保存在 `/root/autodl-tmp/pip-cache`。网络失败时进度不会丢失，重新执行同一脚本即可续传，不要换 URL 或手工删除临时文件。

`setup_autodl.sh` 只建立 Python、vLLM 和 EasySteer 环境，不会下载模型、ReBalance 向量或实验数据。第一次从零安装仍需下载约数 GB；模型和实验资产需要另外复制到文档规定的路径。

## 场景三：已有实例只更新代码

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

数据盘管理：`venvs/`、`wheels/`、`models/`、`hf-cache/`、校准隐藏特征和完整逐题结果。当前MATH训练题、MATH-500、GSM8K数据入口在仓库的`sources/ReBalance/Data/`，具体路径见00。完整克隆系统盘并勾选数据盘后可直接运行；代码落后时拉取，依赖变化需明确核对后处理，不因普通代码更新重装。

两环境必须隔离：`easysteer-vllm026`用于生成，`rebalance`用于原作者回放／判分。`setup_autodl.sh`不等于完整建立这两套研究环境。50GB盘曾接近满，主要大项还包括全层隐藏特征；先核对备份和硬链接占用再清理，不能只按模型大小估算。`/dev/shm`是易失内存盘，不作为迁移备份；已保存答案可用于重新回放特征，详见00的踩坑记录。

环境安装脚本已按本次成功命令合并，但尚未在完全空白的新实例做一次 clean-room 全流程复验；首次队友复现时应保留完整日志，并据此更新脚本。
