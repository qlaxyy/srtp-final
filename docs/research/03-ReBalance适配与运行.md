# ReBalance 适配与运行

更新日期：2026-09-11。本文是当前脚本操作手册；目标、最终指标和版本解释见[00-研究交接.md](00-研究交接.md)。1.5B／7B四组完整结果已冻结，阅读或接手不需要执行生成命令。

## 当前链路与入口

当前基线先用本模型500道独立MATH训练题生成答案，再回放隐藏状态、自动选层、自提向量并校准参数；随后在vLLM中逐请求计算算术平均置信度和两步方差，通过EasySteer注入动态向量。模型权重不训练，作者原码与我们的补全边界见00。

下列路径相对于`integration/rebalance_easysteer/`：

| 入口 | 用途与限制 |
| --- | --- |
| `scripts/run_auto_baseline.sh` | 新模型完整流程：生成校准答案→prepare/collect/select/fit→MATH与GSM评测；已完成模型不要从头执行 |
| `scripts/calibrate_auto.py` | 复用保存答案，完成步骤对齐、decoder层特征回放、按题分组自动选层、向量／参数计算 |
| `scripts/run_auto_baseline.sh --group math500`或`--group gsm8k` | 只评测指定数据集，使用已有校准资产，不重新校准 |
| `scripts/run_final_7b_math500.sh` | 冻结7B MATH配对入口：16000上限、17408上下文、并发64、不限墙钟时间、要求新目录。已执行完毕，再次执行是新测评，不是查看结果 |
| `eval/resume_validation.py` | 由评测入口调用，检查续跑协议、保留完整答案、只补缺题 |
| `scripts/regrade_saved_results.py` | 对保存答案调用作者判分，不重新生成 |
| `scripts/freeze_final_results.py` | CPU检查完整配对、协议／判分匹配及资产哈希，再生成冻结清单；现有冻结JSON不要覆盖 |
| `scripts/audit_calibration_labels.py` | 原30题／76步内容审查：准备、按轮查看、揭示代理标签；样本及旧审读已固定，不重新抽样 |
| `scripts/audit_answer_evidence.py` | 本地CPU答案证据索引、查看与审读校验；不调用模型，不自动分三类，不改向量 |
| `scripts/audit_control_alignment.py` | 把已有30题证据与原函数在前后边界的系数对齐；CPU投影，不是实际干预效果测试 |
| `scripts/audit_repeat_confidence.py` | 读取已存30题／76步内容证据与本步统计，比较重复、新内容和冻结高／低阈值；不重新生成或拟合 |
| `scripts/replay_cycle_monitor.py` | CPU逐token回放原500题、记录精确相邻段落块重复；只报警，不截断、不选择答案、不调用模型 |
| `scripts/replay_atom_cycles.py` | CPU检查段落内相邻片段重复，投影原边界的正向注入机会；只记录证据，不实际干预 |

`run_rebalance_static_vllm.sh`、`run_rebalance_dynamic_vllm.sh`、`run_own_calibration.sh`、`run_paper_baseline.sh`保留历史路径，不能替代当前自动校准基线入口。没有默认先跑20题的要求。

## 环境与运行前检查

服务器仓库`/root/autodl-tmp/projects/srtp-final`。生成使用`/root/autodl-tmp/venvs/easysteer-vllm026`；离线回放／校准／作者判分使用`/root/autodl-tmp/venvs/rebalance`。脚本直接调用环境Python，无需activate，不混装依赖。

vLLM／EasySteer以editable方式指向`sources/EasySteer/`；wheel提供二进制依赖，实际Python控制代码在仓库里。替换成未打补丁的原生vLLM后不能声称仍在运行当前方法。

新实验前核对工作区／提交、磁盘余量、GPU占用、模型及向量参数哈希、新输出目录。仅查阅文档不必加载模型或重新做GPU验收，已有环境不重新安装。

## 通用脚本关键变量

下面是未来明确的新实验的参数说明，**不是本次接手要执行的任务**。

| 变量 | 含义 |
| --- | --- |
| `MODEL` | 模型绝对目录，必填 |
| `CALIBRATION_SOURCE` | 本模型校准生成目录，必填；存在完整`generation_summary.json`时复用，不重生成 |
| `OUTPUT` | 本次输出目录，必填；新实验独立目录，不覆盖冻结结果 |
| `EVAL_ARTIFACTS_DIR` | 只评测时指定已有向量／fit目录，默认OUTPUT |
| `BASELINE_MATH500` / `BASELINE_GSM8K` | 显式复用同协议、完整无干预结果；不匹配应拒绝，不拿部分结果充数 |
| `RESUME_CALIBRATION=1` | 接续未完成校准生成，保留完整题目 |
| `RESUME_RESULT` / `RESUME_ELAPSED_SECONDS` | 评测接续的保存结果及中断耗时，要求协议匹配，新目录保存结果 |
| `EVAL_MAX_TOKENS_MATH500` / `EVAL_MAX_TOKENS_GSM8K` | 均默认16000；对比方法同上限，触顶仍计入统计 |
| `EVAL_MAX_MODEL_LEN` / `EVAL_MAX_NUM_SEQS` | 总上下文／并发；默认上下文按上限＋1024向上对齐512（16000时为17408），并发32 |
| `EVAL_CHUNKED_PREFILL` / `EVAL_MAX_BATCHED_TOKENS` / `EVAL_GPU_MEMORY_UTILIZATION` | 分块预填充／单轮token预算／显存预算；冻结7B采用开启／2048／0.92 |

OUTPUT已有`eval_runtime.json`时，其中上下文和并发会覆盖相关设置；先检查文件，不只看环境变量。当前支持具备对应think／空行token和Qwen／Llama式decoder结构的推理模型，其他结构需要显式适配，不保证任意模型开箱即用。

**超时有两层：** `CALIBRATION_TIMEOUT_SECONDS`默认1500秒，prepare/collect/select/fit各1500秒。`EVAL_GROUP_TIMEOUT_SECONDS`默认1500秒，设0关闭评测／作者判分预算；但完整流水线每个数据集外层仍有硬编码3300秒超时。冻结7B入口直接调用`--group math500`，关闭内部预算且不经过外层限时。新模型若需更长预算，先处理实际入口的两层限制；运行后改设置不能撤销已启动计时器。

## 复用、续跑与冻结

### 1.5B 高利用率配置参考（2026-09-10）

已在本机单张RTX 4090D、现有EasySteer/vLLM环境、DeepSeek-R1-Distill-Qwen-1.5B上完成MATH-500验证。作为同模型其他题集的起始运行配置；不是任意1.5B模型或7B的通用最优配置。题集的提示词／生成长度和排队情况会影响显存、抢占及吞吐，不能把MATH的408.3秒外推到GSM8K。后续明确的新实验可沿用并记录实际统计，不必重复性能短窗。

| 设置 | 参考值 |
| --- | --- |
| 精度／设备 | BF16、单张RTX 4090D |
| 调度 | `--async-scheduling` |
| 并发／每轮token预算 | `--max-num-seqs 128 --max-num-batched-tokens 32768` |
| 分块预填充／前缀缓存 | 均关闭（不传`--chunked-prefill`；入口关闭前缀缓存） |
| 显存预算 | `--gpu-memory-utilization 0.90` |
| 总上下文／新生成上限 | `--max-model-len 32768 --max-tokens 16000` |
| 采样 | `--temperature 0.7 --top-p 0.95 --seed 42` |
| 动态执行 | in-graph、保留KV历史；运行中记录抢占／恢复 |
| 评测墙钟限制 | `--group-timeout-seconds 0`，完整答案逐题落盘 |

运行入口是实验分支`codex/first-step-half-20260910`的`eval/rebalance_dynamic_eval.py`（代码引入提交`1cab764`）；main只保存记录，不能拿main入口直接传候选参数。沿用`/root/autodl-tmp/venvs/easysteer-vllm026/bin/python`，同时把该环境`bin`放在PATH最前并设置`PYTHONNOUSERSITE=1`，避免已有ninja无法被找到。

方法参数与上述运行配置分开：1.5B仍用`auto_code_v2_500_20260908/auto_vector.pt`、`fit.json`、decoder输出层20。首段起点−0.5候选额外传`--first-step-comparison --candidate-only --first-step-coef -0.5`；这些参数不是所有未来方法都要启用的性能设置。新目录单独保存结果，作者判分入口只指定`--group rebalance_dynamic`，数据类型与该题集冻结判分一致。

**两个强度的区别：** `first_step_coef=-0.5`仅在最后一个prompt输入位置注入一次；`initial_coef=-1`是每条请求创建时的控制器状态初值，不是后续每步固定施加−1，也不在每步重新设为−1。首次有内容统计的步骤边界先计算动态强度并覆盖该状态，再用于边界注入；只有尚无有效内容统计的特殊边界才可能沿用初值。对应实验源码`steer_vector_utils.py`中`add_request`初始化与提示词末尾历史赋值，以及`ready = is_boundary & (counts > 0)`后更新系数的逻辑。后续步骤仍按当前置信度和相邻步骤波动计算。

1. 同模型已有500题答案：优先复用；改变离线方法时回放答案并写入新向量／参数目录，不能根据测试成绩反复选参数。
   只改变在线注入时机／初值的对照可复用已有层、向量和其余动态参数，跳过整套校准。先在CPU核对改动是否真正进入注入：当前版本仅在生成分隔token触发，初值可能在首次有效注入前已被动态值覆盖，单改`initial_coef`不等于“关闭首段干预”；9月10日检查及新候选见00。
2. 只缺部分评测：核对协议和完整题数，只补缺题；保留原文件及恢复来源。重新排队可能改变新输出，不宣称等价于连续运行。
3. 生成已完成仅缺判分：只处理保存文本，作者口径与math-verify分开记录。
4. 全部完成：保存逐题文本/token、截断、正确数、协议、资产／源码哈希及运行账本，只汇总完整配对；未来方法另建目录，不覆盖JSON、不移动旧标签。

当前资产位置见00及[最终冻结清单](../../integration/rebalance_easysteer/configs/final_results_20260909.json)。7B旧`formal_kv_replay/`内“MATH未完成”由另一个最终目录补齐，不能据此重新测评。

## 本地CPU答案证据工具

使用Python 3.11以上和标准库，无需服务器或额外安装依赖。默认索引已存在于`.codex_work/label_evidence_30_20260910/v2/evidence_index.json`；在仓库根目录查看Q16：

```text
python integration/rebalance_easysteer/scripts/audit_answer_evidence.py show --first 16 --last 16
```

默认展示检索线索、目标相邻步骤及精确重复的代表；`--all-steps`展示所有不同完整步骤。全文和未完成尾段仍在索引，屏幕长尾预览有限长。分隔token中的数学符号在展示时恢复，`start:stop`保留校准跨度，`start:evidence_stop`是含该符号的取证跨度；均为从0数的生成token半开区间。

`prepare --output <新目录>`从原归档及固定30题重建索引并核验哈希，不生成答案；目标已有索引时拒绝覆盖。`finalize --output <索引目录> --review <审读JSON> --report <新结果JSON>`核对引用、前缀位置、精确有理数及人工修订后汇总。当前审读输入是上述本地目录的`answer_anchor_review.json`，并嵌入[结果摘要](../../integration/rebalance_easysteer/configs/calibration_label_evidence30_20260910.json)的`questions[].annotation_input`。检索词和精确重复只帮助找证据，不是语义判定规则；未找到见证不自动标错或欠思考。既有摘要无需重新生成。

## 本地CPU控制强度对齐工具

本轮已完成，直接查看[结果摘要](../../integration/rebalance_easysteer/configs/calibration_control_alignment30_20260910.json)，不必重复运行。需要重新核对新的工具改动时，使用Python 3.11以上及NumPy；本地现有运行时已具备，不需要安装torch、加载模型或启动服务器：

```text
python integration/rebalance_easysteer/scripts/audit_control_alignment.py --output <新目录> --report <新结果JSON>
```

输入依赖原校准归档、固定30题的`audit_protocol.json`／`private_key.json`、答案证据索引、原`curve_check.json`和仓库内冻结清单／1.5B参数。按文件哈希验证后，从当前冻结源码提取数值函数，在CPU重建4652步系数；不import vLLM。`incoming`是前一步边界可能施加的系数，`outgoing`是本步结束后的系数；首个prompt和结束思考记无注入边界。float32模拟仅核对数值敏感性，不代替CUDA运行验证。

默认完整产物目录为`.codex_work/control_alignment30_20260910/`，其中`projected_steps.json`保存全部步骤，已纳入Git的结果摘要保存76个检查位置和44条答案见证的对齐。新输出不得覆盖已存在文件；该工具不提供续写、调参或修改在线控制器的入口。

9月11日另已完成[重复与置信度核对](../../integration/rebalance_easysteer/configs/repeat_confidence30_20260911.json)。它直接读取上面的已存投影及证据索引，使用标准库，不需要重跑控制对齐。确需核对新改动时，用`scripts/audit_repeat_confidence.py --output <新目录> --report <新结果JSON>`；路径仍相对`integration/rebalance_easysteer/`，默认产物位于`.codex_work/repeat_confidence30_20260911/`。规则在代码提交`1eed285`中预先固定，任一输出已存在即拒绝。注意这次统计本步结束后的系数，不能与旧交接中本步之前的计数混用；分层样本比例不是总体误判率，逐字相同历史也不证明重复导致置信度上升。

## 本地CPU循环回放工具

9月11日首轮500题已完成，直接查看[回放结果](../../integration/rebalance_easysteer/configs/cycle_monitor500_20260911.json)，不用重跑。工具仅需Python 3.11以上与标准库；复用原校准归档、现有tokenizer、冻结参数及30／20题样本清单，无需服务器、torch或重新生成。需要验证后续工具改动时，在仓库根目录指定**新目录**：

```text
python integration/rebalance_easysteer/scripts/replay_cycle_monitor.py --output <新目录>
```

默认方案为`configs/cycle_monitor_plan_20260911.json`，运行前已固定并提交。输出目录存在即拒绝覆盖；先落盘方案，再核对输入哈希，逐token记录已闭合段落的重复报警。`summary.json`含500题记录、停止状态交叉统计、首报位置和固定抽查索引；`details.json`另存全文、全部段落和重复块。已完成目录是`.codex_work/cycle_monitor500_20260911/`，其中`review/`保留本次审读窗口及哈希，Git结果摘要记录全部500题位置和18份审读证据。

生成token位置按从1数的数量记录，跨度为从0数的半开区间；审读段号从1数，校准索引／训练索引从0数。这种字节流段落划分用于取证，不替换原校准步骤对齐。它会漏掉未闭合段落内的循环，正常结束也可能报警；报警后剩余token不是实际节省量，停止状态不是思考类别金标准。当前工具没有GPU、自动结束、注入强度或阈值搜索入口。

## 本地CPU段内重复与注入机会工具

500题回放已完成，直接查看[结果摘要](../../integration/rebalance_easysteer/configs/cycle_atoms500_20260911.json)，不必重跑。使用Python 3.11以上与NumPy，本地现有运行时已具备；不需要torch、GPU或服务器。确需核对后续工具改动时，指定**新目录**：

```text
python integration/rebalance_easysteer/scripts/replay_atom_cycles.py --output <新目录>
```

方案为`configs/cycle_atoms_plan_20260911.json`，运行前提交为`892f9dc`。输入复用原归档、tokenizer、1.5B参数／冻结源码及已完成的段落回放结果；旧段落规则不重新运行。输出目录存在即拒绝覆盖。完成目录`.codex_work/cycle_atoms500_20260911/`内`summary.json`保存500题统计，`details.json`保存全文、重复记录和正向机会，`review/`保存有范围说明的审读窗口；Git摘要另含审读结论及原始文件哈希。

匹配保留完整单词、数字与符号，仅忽略空白，未闭合元素不在结束时强行补齐。它可以匹配正常公式内部的短重复，因此不是语义思考状态分类器。系数投影沿用原算术置信度／两步方差，仅在原边界且片段仍重复时统计正系数；`1e-6`是近零数值报告容差，不是循环阈值。机会数来自无干预轨迹，尚未实际施加`min(原系数, 0)`，也没有新建注入位置、测量在线开销或证明压缩收益。

## 完整步骤正向开关（实验分支，尚未GPU验收）

检测实现位于`codex/repeat-positive-gate-20260911`／`8df7c2a`，配对入口与固定100题在`89160b1`，运行前检查修正为`a61b590`，main仅保存[CPU检查结果](../../integration/rebalance_easysteer/configs/repeat_gate500_20260911.json)。本地checkout是`.codex_work/repeat_gate_worktree/`。代码入口为该分支的`eval/repeat_positive_gate.py`，由`eval/rebalance_dynamic_eval.py`显式安装；没有修改EasySteer源码、冻结向量或动态函数。

默认`--repeat-gate off`完全不安装桥接；`shadow`只检测并记录原施加强度，`cancel_positive`在完整相邻步骤块重复时裁剪当次实际施加强度为不大于0，保留原控制器状态。两种启用模式都增加CPU读取采样token的同步开销，不能把9.49秒离线回放当作在线开销。当前仅允许`--diagnostic-group rebalance_dynamic`、已有`--calibration-fit`、`--max-tokens 16000`的新工程单组运行，拒绝混入论文重建版、断点续跑或性能剖析；单组产物仍标为工程结果，新的配对包装器只有在两组各100份完整、协议一致时才组合为训练验证记录。入口新增`--async-scheduling`，默认仍同步；固定100题方案明确启用异步。尚未启动GPU。

CPU回放已经完成，不必重跑。确需核对新实现时，在**该分支checkout**中运行：

```text
python integration/rebalance_easysteer/scripts/replay_repeat_gate.py --assets <保存原归档与历史结果的主仓库目录> --output <新目录>
```

使用现有Python与NumPy即可，无需服务器或torch。`--assets`可以指向主checkout以复用其`.codex_work`原归档；输出必须新建。完成目录`.codex_work/repeat_gate500_20260911/`中保存方案、500题摘要、全部边界／步骤、10个审读窗口及范围说明。线上桥接要求单进程V2 runner与施加强度历史，KV抢占同时保留检测器的未完成片段、完整步骤和观察token数；CUDA及真实抢占行为尚未验收。输出摘要另记tokenizer／桥接源码哈希、原／实际强度、改变位置和复制批次数。


固定100题已准备，无需重新运行抽样器。**在实验分支checkout内**执行下面的只读预检，不会导入torch、连接服务器或生成答案：

```text
python integration/rebalance_easysteer/scripts/run_repeat_validation.py --bundle integration/rebalance_easysteer/configs/repeat_validation100_20260911 --output <未来的新结果目录>
```

GPU方案经用户确定后，服务器同一命令显式增加`--execute`，输出建议为`/root/autodl-tmp/results/easysteer/repeat_validation100_20260911`。包装器沿用两个现有环境，确认现有环境实际导入路径属于当前实验checkout（否则拒绝），核对干净Git、35个冻结源文件（仅评测入口按实验版哈希替换）及新增实现、模型5文件、向量、fit和题目哈希；源文本统一LF校验，题目文件固定LF并用原始字节哈希。两组以独立进程、同一1.5B异步配置运行，原动态完全关闭检测，候选包含检测同步开销。没有默认额外20题或合成吞吐试验。

运行输出为`original_dynamic.json`、`repeat_cancel_positive.json`及各自日志／逐题partial，完整后生成`paired.json`、`author_grading.json`、`analysis.json`和`run_ledger.json`。为了沿用作者判分器，组合文件的`baseline`键明确表示**原动态版**，`rebalance_dynamic`键表示候选，不是无干预对照；协议保留两份原单组信息与对应含义。中途异常标记incomplete，不自动重启／覆盖已有结果；如果只欠判分，使用保存的`paired.json`单独判分，不重跑两组。完整固定方案、题号与判断规则见00及[验证准备记录](../../integration/rebalance_easysteer/configs/repeat_validation100_20260911.json)。

## 单边界配对续写（实验已完成）

20题／40份完整结果见00与[结果摘要](../../integration/rebalance_easysteer/configs/boundary_ablation20_20260910.json)，无需重跑。代码只在`codex/boundary-ablation-20260910`，当前main保留计划和结果；进入实验代码前检查Git状态和目标提交。正式生成提交`921793f`，判分兼容修正提交`14fe410`。

- `scripts/prepare_boundary_ablation.py --output <新目录>`：CPU从已存500题中固定20题和边界，排除原30题，输出`manifest.json`和不含标准答案／未来token的`model_inputs.json`。现有计划已固定，不重新抽样。
- `eval/boundary_ablation.py run --prepared <已核验输入目录> --output <新目录>`：在原EasySteer环境运行40份接续，向量默认沿用`auto_code_v2_500_20260908/auto_vector.pt`；运行前将该环境bin置于PATH首位并设`PYTHONNOUSERSITE=1`。输出逐条落盘，异常保留部分记录；当前入口没有自动补缺或续跑功能，异常后不能直接重复完整批次。
- 同一入口`grade --prepared <输入目录> --output <已完成运行目录>`：使用现有ReBalance Python 3.10环境进行作者判分，不调用模型，拒绝覆盖已有摘要。前缀加新增token一起判分／统计，16000是二者合计上限。

接续专用参数为`prefix_mean`、`prefix_variance`、`prefix_apply`，经API与请求结构传入状态管理；不是重新设置首步常数。历史前缀不注入，仅将最后一个已完成边界纳入选择器；两组恢复相同的前一步均值／已计算系数，B只把这一个位置的实际掩码设0。之后沿用原动态更新，KV抢占时恢复历史掩码及状态。实际掩码按BF16精确核对，与控制器保存的float32系数分开记录。缺省不传这些参数时，原prompt起点行为保持不变。
