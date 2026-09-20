# 完整在线校准采集：固定范围

仅准备，本目录尚无GPU结果。现有8题工程门槛不代表完整采集已经通过。

## 目的与边界

用原生受干预模型在原MATH训练校准500题上采集步骤首token的decoder20（hidden21）pre/post状态和实际在线pmax。两来源为冻结单R、冻结R+L27。正确性不参与步骤筛选；既有原无干预500题不重新生成，冻结测试集不运行。完整采集不是效果实验，不比较本次吞吐与旧生产结果。

测量层固定为旧向量所在层，暂不重新自动选层。pre状态作为主要分析对象，post保留作诊断；二者均受到过去干预/KV的影响。单次局部pre/post差异不能估计完整干预的因果作用。若要对比“同一新轨迹上的无干预回放”，必须另作匹配回放，不能拿历史不同生成前缀冒充对照。

后续只比较两个预定标签规则：A保留原O/U；B只将触顶轨迹排除出U，O不变，置信度/方差分位数仍用全部有效步骤计算。不额外提高分位数、不新增低方差条件、不按正确性筛选、不同时加入按题均衡。先报告步骤数、题数、触顶/错题来源、前10题贡献、端点差异，再决定拟合；本批runner不会提取新向量或跑验证。未触顶不是欠思考真值。

## 采集与成本

1. 稀疏工程检查：相同原训练前8题，R/L27各一组、各512上限，共16份/8192token。与上次已保存的dense工程结果比较token、native pmax/控制记录和所有被保留的pre/post状态，要求精确相同。改变batch容量32→64若影响数值也会使门槛失败，不强行解释为等价。预计1–3分钟，硬上限900秒。
2. 通过后完整采集：1.5B/BF16，原训练500题×R/L27两组，共1000份；seed42、greedy、top_p1、cap16000。每64题固定一批，最后52题；seq64、batchedtokens32768、GPU .90、TP1、eager/split同步、无prefix cache/chunked prefill/推测解码。每批严格禁止槽位复用，批间重置全部观察器和L27状态。预计20–40分钟，完整采集进程硬上限3600秒；最大1600万生成token，计时超限停止，不追加预算。

每128次forward合并转CPU，只保留步骤首token的pre/post数组；pmax与轻量控制记录保留全部。最多暂存128×64×1536×两份BF16约48MiB，加上暂态prefill快照/CPU转换，需另预留内存。原8题×2的8192位置，经CPU实测仅保留184个步骤首位置，精确匹配。长轨迹稀疏程度可能不同，不把此比例当全量保证。预计输出约1–2GB；开始要求数据盘至少3GB余量，每批结束检查余量，不自动删除历史数据。

完整collector保存全部候选首位置，最终分析必须用原spans规则排除think结束后与未闭合末步骤；最后生成token没有被再次输入时不能伪造隐藏状态。32位保存BF16值，不恢复已丢精度。受控轨迹若与历史答案不同，要按本次文本重新判分，不能套用旧标签。

## 运行

沿用当前环境，不安装升级、不改共享代码。包按目录结构解至独立`/root/autodl-tmp/projects/online_endpoint_full_20260919`。先运行本目录`run_sparse_gate.py --runtime-root /root/autodl-tmp/projects/hybrid_wsc_native_long_20260917 --output /root/autodl-tmp/results/easysteer/online_endpoint_full_20260919_gate_run1`。通过后运行`run_collection.py`，同一runtime-root，`--sparse-gate`指向该门槛目录，`--output /root/autodl-tmp/results/easysteer/online_endpoint_full_20260919_collect_run1`。Python为`/root/autodl-tmp/venvs/easysteer-vllm026/bin/python`，PATH沿用venv/bin、CUDA/bin和标准系统路径。

输出必须不存在；失败重试独立run_id、不拼接成完整正式结果。哈希不符、前缀/位置错位、非有限数、槽位复用、门槛不一致、超时或空间不足均立即停止。保留已完成批及partial，设备异常可能无法导出未完成的GPU缓存。成功回执必须列明1000份完整身份和产物哈希；GPU结束后退出。此任务不授权7B/GSM8K/新测试集或第二轮迭代。
