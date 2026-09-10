# ReBalance 适配与运行

更新日期：2026-09-10。本文是当前脚本操作手册；目标、最终指标和版本解释见[00-研究交接.md](00-研究交接.md)。1.5B／7B四组完整结果已冻结，阅读或接手不需要执行生成命令。

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
