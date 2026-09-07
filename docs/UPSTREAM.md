# 上游来源

本仓库为研究复现整合仓库，不是两个上游项目的官方镜像。

| 目录 | 上游 | 固定版本 |
| --- | --- | --- |
| `sources/EasySteer` | https://github.com/ZJU-REAL/EasySteer | `b771d3104f7f7c3181e7ce3b50fab565f8d08282` |
| `sources/EasySteer/vllm-steer` | https://github.com/ZJU-REAL/EasySteer-vllm-v1 | 基线 `6267ca0cfc9c6e93b1427d36b1d655821d6d6f9b`，已合入本项目动态 ReBalance 修改 |
| `sources/ReBalance` | https://github.com/yu-lin-li/ReBalance | 初步复现源码快照；2026-09-07查得上游 HEAD 为 `c7207ee1583e42170ac4f324c12345863b3d40f0`，两份重点动态推理源码与在线上游文本一致；全目录尚未逐文件比对，不能认定整个快照已精确锁定 |

各目录保留原许可证。以后同步上游时必须单独提交，并记录新的上游提交和本项目回归测试结果。
