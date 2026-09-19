# 一轮迭代重校准：R 与 L27 两条来源

状态：仅本地准备。尚未连接服务器、运行GPU、生成新答案或产生新效果结论。
基准提交：45f0334857d5f2bc6555eedfa845fa00cecc5bbd。使用原B线worktree，冻结标签不变。

## 研究对象与理论边界

本版检验：干预改变生成轨迹分布之后，在这些新前缀上重新校准基础模型，是否改善下一轮压缩。
两条来源是原500道MATH训练题的R与L27输出；不配长短偏好，不筛正确或更短的题，不引入外部教师。
L27已有500题完整保存轨迹直接复用（历史404题seq256+96题seq32恢复，完整保留来源），只新生成R的500题。
新生成使用原prompt IDs，greedy、seed42、cap16000。全部错误和触顶纳入。

关闭干预回放是本版明确选择，不声称理论唯一正确。它测量基础模型在新前缀下的状态和raw pmax。
带干预回放测量上一轮策略的状态，适合研究“现有策略上的残差纠偏”；不能未经说明把这种差值替换基础模型向量。
单个注入点上，若h'=h+alpha*v，则两组均值差额外包含(mean(alpha|O)-mean(alpha|U))*v；深层和缓存还会有非线性历史效应。
因此本版同时关闭R和L27重算置信度及状态，不混用生成时置信度与基础模型状态。
这也不同于最初无干预greedy的直接logprobs：新轨迹上必须算full-vocabulary pmax，不能把已保存token的概率当作pmax。
HF BF16 SDPA与vLLM有已知数值差异，不宣称逐token分布等价；这属于本版测量口径限制。

## 固定处理规则

每个来源独立：保留原token IDs；以含双换行的token为边界；自然think结束计入最后完整步骤；触顶轨迹末尾未完成步骤不计，先前步骤仍保留。
置信度是步骤内每个预测位置raw pmax的算术平均；方差是相邻两个步骤置信度差平方/4，首步为0。
过思考候选=大词表命中 OR c<本来源Q25；欠思考候选=未命中 AND c>本来源Q75。
这些是原方法代理标签，不是逐步骤真实错误/冗余真值。
原隐藏状态位置为步骤第一个token的decoder block输出；full replay中的置信度来自其前一位置预测。
提取所有层，复用原calibrate_auto.select的按题80/20层选择与fit实现；重新计算raw均值差、LDA目标、c/v分位数和曲线参数。
不匹配旧向量范数、不固定旧层、不加新旧向量、不更改词表、函数形式或低/高置信度标签规则。
此版一次更新，不是已经闭环运行多轮的自迭代。

## 第一批GPU：只收集与校准

- 模型：DeepSeek-R1-Distill-Qwen-1.5B，BF16。
- 新生成：原MATH-train校准500题，单R一组；另8题×2次×最多512 token仅作工程检查。
- 已有轨迹复用：L27的500题，无新生成。
- 回放：每来源4个工程前缀×2次；通过后每来源500份完整思考前缀，一次前向同时读各层和raw pmax。
- 生成vLLM参数：TP1、in_graph、async、max_num_seqs256、max_num_batched_tokens32768、显存比例0.90、无prefix cache。
- 回放为Transformers前向，不是vLLM生成；不许为了报GPU利用率而把两者混在同一耗时中。
- 粗估：生成5–20分钟，两组回放合计10–30分钟，另有CPU选层/拟合与加载开销。仅估计，不能沿用MATH测试集378秒作为训练500题预算。
- 硬上限：生成工程600秒/完整2100秒；每来源回放工程600秒/完整2400秒；超时保留partial，不自动降batch拼接或延长预算。
- 记录生成时间、回放/拟合时间、前向次数、GPU监控、模型和资产hash。GPU占用冲突则停止。

命令模板：以下在保留仓库目录结构的独立部署目录执行，PY指向现有easysteer-vllm026环境；不安装环境。
ITER_HOME指向本目录的远端副本，OUT固定为/root/autodl-tmp/results/easysteer/iterative_recalibration_20260919_run1，RUNTIME指向已核验运行时。

```bash
$PY "$ITER_HOME/generate.py" --runtime-root "$RUNTIME" --output "$OUT/R_engineering" --engineering
$PY "$ITER_HOME/generate.py" --runtime-root "$RUNTIME" --output "$OUT/R_train500" --engineering-result "$OUT/R_engineering"
$PY "$ITER_HOME/recalibrate.py" --source "$OUT/R_train500/result.json" --output "$OUT/R_replay_engineering" --engineering
$PY "$ITER_HOME/recalibrate.py" --source "$OUT/R_train500/result.json" --output "$OUT/R_fit" --engineering-result "$OUT/R_replay_engineering"
$PY "$ITER_HOME/recalibrate.py" --source "$ITER_HOME/saved_l27.json" --output "$OUT/L27_replay_engineering" --engineering
$PY "$ITER_HOME/recalibrate.py" --source "$ITER_HOME/saved_l27.json" --output "$OUT/L27_fit" --engineering-result "$OUT/L27_replay_engineering"
```

所有输出目录必须首次创建。

## 第二批：新资产核验后两组完整MATH-500

先发布两份实际fit/vector的SHA256、层号、类样本数、向量范数与夹角及曲线检查；当前没有这些资产，不伪造可执行评估release。
R来源的新校准搭配单R；L27来源的新校准搭配L27。每组MATH-500全500题，seed42、T=.7、top_p=.95、cap16000，其余原运行设置。
每组先用8个固定工程题确认资产加载、重复执行、状态清零、输入位置和层号；既有评估runner将通过独立release引用新资产，不复制控制器。
冻结R/L27全量结果作为各自主要参照，并同时列对方结果。复用前核验prompt/模型/采样/判分/运行身份；若不兼容，停止说明，不能悄悄把旧基线当同批配对。
本次MATH-500已曝光，只是开发比较，不能称独立确认，不能按500题反复调阈值。
预期两组纯生成15–25分钟；完整进程预算1800秒/组。此处是后续固定计划，第一批成功后再形成带真实资产哈希的运行release。

固定晋级点值：对各自旧版本，思考token和总token均下降、正确率损失不超过2个百分点、触顶不增加；全部报告，不只报告胜者。
报告准确率、思考/总token、触顶、WC/CW、纯生成时间、额外前向/回放费用，逐题配对bootstrap95%区间；小幅收益不设5%门槛，区间跨0则未确认。
优于单R不等于优于L27；点值非劣不等于统计确认2pp界。失败不启动第二轮，不追加seed、GSM8K或7B挽救。

## CPU检查和未完成事项

已检查500个唯一题号/题面hash、旧L27归档身份与500题完整性、4个边界用例、Python语法、输入与依赖hash。
现有GPU环境未在本次加载；不能把CPU通过写成GPU已跑通。第一批工程门槛尚待GPU执行。
原U500答案不重生成，旧L27不重生成，A线资产与共享vLLM控制器不修改。校准题为已登记共享训练资产，没有新增独立题划分。
