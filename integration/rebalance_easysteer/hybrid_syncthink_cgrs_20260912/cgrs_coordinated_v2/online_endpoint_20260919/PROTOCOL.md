# 在线端点诊断：GPU前固定协议

目的：分开检验测量对象和端点筛选。当前不拟合新向量、不生成完整测试答案，不声称筛掉步骤就能提高指标。

CPU筛选只使用原校准500题。高置信度90/95分位、低方差、排除触顶及两者交集仅做数量/集中度诊断，不按测试集正确率挑参数。正确性只统计，不参与筛选。暂保留“非反思词且c>q75，所在轨迹未触顶”为下一候选；O标签不变，q75不改，不同时加低方差或按题均衡。未触顶不等于真正欠思考，保留此证据缺口。

新增状态采集用原生vLLM实际生成。模型1.5B/BF16，原校准前8题，四臂R_off/R_observe/L27_off/L27_observe；seed42、greedy、top_p1、每份512上限，总32份/16384 token；过程硬上限900秒，预计3–8分钟，硬上限15分钟。这是工程数据，不判压缩或准确率，截断末步骤不可参与拟合。数据身份见rows.json及plan中的哈希。原500题均为已曝光校准资料，不新增独立确认划分。

原冻结向量、函数、词表不变。运行工程检查使用eager、同步、TP1、seq32、batchedtokens32768、显存.90；这是为了明确捕获原生层hook，不作为吞吐结果。不能把此模式采集与旧async/CUDA-graph轨迹声称逐token等价。后续生产模式采集需单独验证，不自动扩展。

测量decoder20（hidden21）原生residual stream：prepend层输出hook在已有steering hook之前复制pre；追加hook复制post；下一层输入检查post精确一致。两者均保留，主分析优先pre，因为直接添加向量造成的位移不应被当成新学到的方向；pre仍包含此前干预与KV的影响。post用于诊断，不能未经证明减去alpha*v冒充pre（BF16舍入和实际应用位置均重要）。记录全部工程token，在CPU上按prompt长度映射步骤首token输入位置；不是生成该token的前一位置。末token无下一次输入状态时不得伪造。

NativeAudit复制observe_sample实际接收的完整词表pmax，发生在L27惩罚和温度/top-p处理之前，但包含R对本次模型前向的影响。两臂均开启这一最小审计，比较token、pmax、coef、上一完成步骤均值、当前步骤计数。隐藏状态观察器开关必须完全一致。无槽位复用、推测解码、异步、prefix cache或chunked prefill。

停止条件：哈希不符、GPU被其他进程占用、位置/残差不一致、非有限状态、捕获溢出、开关输出或控制状态不一致、没有实际向量干预覆盖，任一发生即停止并保存已完成臂和partial；不据此运行全量。异常设备状态下可能无法导出未完成的GPU快照，failure.json和逐题partial仍保留。

实现仅新增本目录，复用既有WSC观察器的数据对齐部分与冻结L27适配器，不修改共享控制器/环境。默认关闭隐藏状态观察器不装hook。原始答案、冻结标签和旧结果不改。

运行环境沿用/root/autodl-tmp/venvs/easysteer-vllm026/bin/python，runtime-root为/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917。将package.tar.gz解到新的/root/autodl-tmp/projects/online_endpoint_20260919目录后执行：

```bash
export PATH=/root/autodl-tmp/venvs/easysteer-vllm026/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
/root/autodl-tmp/venvs/easysteer-vllm026/bin/python /root/autodl-tmp/projects/online_endpoint_20260919/integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/online_endpoint_20260919/run_engineering.py --runtime-root /root/autodl-tmp/projects/hybrid_wsc_native_long_20260917 --output /root/autodl-tmp/results/easysteer/online_endpoint_20260919_engineering_run1
```

输出目录必须不存在，重试另取run_id。数据量预计小于0.3GB，额外GPU状态缓存约0.15GB以内；不保存全词表logits、不额外做答案探测。本批不需要重跑原无干预500题。
