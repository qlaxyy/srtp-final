# 原回放状态：置信度上分位数75%→90%

用户核对截图7530后要求继续并开启GPU。本轮回到截图的原L27轨迹及关闭干预回放状态，不使用online_endpoint的另一批轨迹。单R来源同步处理。

复用已哈希核验的calibration_evidence.tar.gz及selected_features.tar.gz。原q75向量由这些状态重算逐元素精确一致。原自动选层不变（R hidden21/decoder20，L27 hidden23/decoder22），没有重新选层。O仍lexical OR c<q25；U仅把nonlexical AND c>q75改为c>q90。动态拟合同样使用q90高端锚点；方差q25/q75、控制函数结构与自适应tau规则不变。未加入长度、触顶、正确性筛选。

R O28124不变，U9655→5091；L27 O16250不变，U7530→3240。新旧方向余弦R0.992625、L270.974508。两候选三曲线锚点及125751点有限性检查通过。没有新训练生成或回放前向。

## 固定评测

- 模型1.5B/BF16，MATH-500完整500题×R/L27候选两组，1000份输出。
- 同旧评测：seed42，temperature=.7，top_p=.95，max_new_tokens16000；vLLM in_graph/async，seq256，batched_tokens32768，显存比例.90；TP1，prefix cache/chunked prefill关闭。
- 复用已验证evaluate.py主体，改变独立资产/输出路径，移除重复短窗工程运行要求，改为CPU曲线与资产哈希门槛。修改差异已检查。没有修改共享控制器或服务器环境。
- 第一参照是各来源原q75重校准完整MATH500；冻结R/L27为补充参照。原结果作者判分已存入historical_compact，源归档SHA256固定。不是同进程同期对照，不能宣称排除全部批次漂移；输入、代码、资产来源及运行配置核验记录。
- 错误/触顶全部计入，报告正确率、思考/总token、触顶、WC/CW、20,000次逐题配对bootstrap及生成时间。计时含输出收集/逐题保存，不当成纯kernel时间；没有新增probe/模型前向。
- 事前点值门槛：相对对应q75思考/总token均下降，准确率损失≤2pp，触顶不增加。区间单列；点值通过不等于统计确认。MATH500已反复曝光，属于探索，不是独立确认。
- 预估生成15–25分钟，每组进程1800秒上限，失败保留部分输出停止；不自动重跑/扩到其他模型或数据集。

## 路径和版本

本地完整CPU工作目录B线`.codex_work/base_replay_q90_20260920`。远端部署`/root/autodl-tmp/projects/base_replay_q90_20260920`；唯一输出`/root/autodl-tmp/results/easysteer/base_replay_q90_20260920_run1`。日志`/tmp/base_replay_q90_20260920_run1.log`。

原依赖目录`/root/autodl-tmp/projects/iterative_recalibration_20260919/.../iterative_recalibration_20260919`只读复用；运行源码`/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917`全部登记哈希核验。包SHA256 `05f40884305c30130ca73c9c84d702e738fe3bbe740e6a0d2936f728291719c6`，本地/远端一致。启动前4090D空闲，数据盘5.0G可用。

本目录与under_q90_20260920（native状态候选）是不同实验，后者未生成评测，不能混用拟合或结论。冻结标签、原结果未改变。
