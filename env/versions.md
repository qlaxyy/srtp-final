# 已验证环境版本

以下组合已在单张 RTX 4090 D（24 GB）、Ubuntu 22.04 上跑通：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.11.0+cu129 |
| torchvision | 0.26.0+cu129 |
| torchaudio | 2.11.0+cu129 |
| torchcodec | 0.16.0+cu129 |
| EasySteer/vLLM fork | 0.1.dev18960+g6267ca0cf |
| vLLM 基线提交 | 6267ca0cfc9c6e93b1427d36b1d655821d6d6f9b |

驱动版本 550.78 可以运行该组合。环境安装以 `scripts/setup_autodl.sh` 为准。
