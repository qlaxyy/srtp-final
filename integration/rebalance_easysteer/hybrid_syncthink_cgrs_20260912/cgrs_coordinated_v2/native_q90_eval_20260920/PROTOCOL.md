# 第二种测量方式：native状态q90效果评测

用户要求继续测试第二种方式，并在向量分组和动态拟合中用90%收紧。已完成本地CPU准备；20403端口拒绝连接，本批尚未上传/启动。不得把准备完成称为GPU测试完成。

使用under_q90_20260920已经拟合的4套资产，不重新生成原训练500题、不做模型回放或采集。native状态来自真实受干预逐token生成，但采集执行为eager/split且轨迹不同于历史生产批，不能称两种测量方式的严格同轨迹因果对照。本轮比较的是同一native状态上的q75和q90；固定hidden21/decoder20，无重新选层。O、词表、方差阈值、动态公式均不另改。q75c字段在q90资产中明确存q90数值。

GPU批次：1.5B/BF16，完整MATH-500，R_q75、R_q90、L27_q75、L27_q90四组各500题（2000份）。q75同状态版本此前没有生成测试，因此不能拿历史base-replay q75替代。历史冻结R/L27和base-replay q75作为补充参照。每题seed42，T=.7/top_p=.95，cap16000；in_graph/async，seq256、batched_tokens32768、.90显存比例。沿用上一批已跑通的eval主体，无共享控制器或环境修改。

CPU验证：四套向量与fit哈希匹配已存summary；向量分组阈值=拟合高端锚点，曲线有限性已通过；500题清单复用上一批。使用已验证prepare产物，package SHA256见cpu_receipt.json。脚本语法已由ast解析通过，尚无新的远端执行验证。

运行目录`/root/autodl-tmp/projects/native_q90_eval_20260920`；独占新产物目录`/root/autodl-tmp/results/easysteer/native_q90_eval_20260920_run1`。包本地B线`.codex_work/native_q90_eval_20260920/package.tar.gz`，含所有新资产和runner；run.sh另传。已部署的iterative_recalibration支持代码只读引用，runner在加载模型前核验其登记依赖及模型哈希。启动前检查GPU空闲/磁盘，拒绝覆盖目录。

启动使用`timeout --signal=INT --kill-after=30s 3600 bash run.sh`，整批60分钟上限，每组1800秒上限。预计25–40分钟（含加载及判分有波动）；任何错误停止，保留partial、failure及exit_status，不自动重跑或扩大。脚本每组结束即CPU作者判分，q90判分显式读取对应同状态q75已完成标签作逐题比较，校验相同来源、同case和完整500题。

事前门槛：相对同来源native q75，思考/总token点值都下降，准确率损失≤2pp，触顶不增加；报告20,000次逐题配对bootstrap95%区间及WC/CW。点值通过不等于统计确认。相对冻结L27另列实用收益，不能只胜过弱参照就替换冻结方法。记录生成窗口（含输出收集/落盘）、进程墙钟、GPU利用率，probe和额外模型前向均0。MATH500已曝光，结果属于探索，不作独立确认。失败不扫描更多分位数/扩展GSM8K或7B。
