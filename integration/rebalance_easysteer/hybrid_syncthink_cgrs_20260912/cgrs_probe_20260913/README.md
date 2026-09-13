# ReBalance × C-probe v1 实现

本地实现完成，CPU合同检查通过；**原生Torch检查、vLLM执行、GPU回放门槛和效果测试尚未运行**。没有连接服务器、安装依赖、使用新题或重新校准。不是官方CGRS原样复现，不继承S64/mix05的收益或失败结论。

## 方法身份

参考作者仓库 `HuangJiameng/cgrs@27077bbca5a4b08fd3f6503d9e87b5fba7facddd`。保留试答置信度→概率禁词机制，按论文§3.2式(1)使用试答原始全词表熵，§3.3式(2)使用 `p=max(0,(C−0.9)/0.1)`。没有复制作者主程序；没有保留作者top5部分熵、`think_ratio=.6`或分段拼接主路的方法。这些差异必须随实验报告，不能改名为官方复现。

本线的固定设计选择：每新增至少1024个思考token后的首个原ReBalance空行边界探测，最多8次，每次最多32个token，贪心试答；提示为 `\n**Final Answer**\n\boxed`。只对首个完整非空boxed内部的完整token算熵；跨括号token不拆分计算，无完整有效答案设p=0。试答生成首个完整框后立即停止；支持嵌套括号。最终答案置信度不是正确性认证。

禁词仅限原步骤边界后的首个非空白内容位置；空白可跳过，遇到其他正文即失效，下一个边界清空置信度。含空行及后续正文的复合token沿用R的边界，但不当作C的词法起点。14个词ID逐一绑定原1.5B分词器，拒绝不匹配或多token替代。仅改指定词的logit，保留原温度/top-p采样；“But”等仍可能是必要纠错，局部限制不保证无误伤。不强制结束，不加入S64或mix05。

## 接口与生命周期

- `policy.py`：无需模型依赖的状态机、概率、独立按题/位置随机数、框内熵和预算规则。
- `backend.py`：只给本次引擎实例安装采样器代理与请求接入回调，不修改任何共享vLLM文件，也不复制控制器。实际接入`runner.sampler`、`runner.add_requests`、`scheduler.running`及原`SteerVectorState`历史字段；精确基准哈希见`identity.json`。生产代码的私有接口有版本风险，校验不符拒绝运行。
- `run.py`：可执行的5组工程／3组筛选入口，模型、校准和题目身份检查、失败保存、纯生成与探测成本记录。
- `prepare.py`：只接收执行对话已经统一登记的训练题和协调凭据，生成唯一运行计划；不自行选题或伪造授权。
- `grade.py`：沿用既有作者判分环境，记录错变对/对变错及配对bootstrap，不覆盖旧结果。

执行顺序：R隐藏状态注入→原始logits→R先保存原max概率→C按此前试答结果屏蔽触发词→原temperature/top-p→原采样器只采一次→R观察实际采样token→C观察已发布token。C的Bernoulli随机数不消费主采样随机流。

达到检查点后暂停全部已接入的主请求，保留其KV、采样器槽位和R控制器。试答以**原始prompt token IDs＋已接受输出token IDs**为前缀，复制每位置的历史R系数，按原prompt分界重新前向；不是拼接文本重新分词。试答前缀保留原向量注入，试答提示和后代关闭新的R/C注入。主路不会接收试答token或概率。

这是**带历史干预的独立前缀回放分支，不是零成本KV克隆**。每次试答后检查主路token/控制器状态哈希未变。工程阶段另生成同前缀单token审计分支，与恢复后的主路原始logits比较：最大绝对差≤0.02、相对L2≤0.002、argmax相同。此为本线独立门槛，不修改A线BCC门槛；失败不得放宽后继续同批。Roff、Rshadow还必须与R逐token及历史系数哈希一致，且实际发生过回放检查。

状态按请求ID持有，结束思考不可重启；完成/取消回收分支槽位；异常保留主请求以便清理，写出未完成的主token、策略状态、探测熵及回放检查。抢占、异步批队列、多卡、前缀缓存、推测解码不在v1支持范围，明确拒绝；未知状态不降级为重建文本。

**v1使用同步调度**，仍用BF16、TP1、256槽位、32768上下文/批token、CUDA Graph、显存0.9，正式主链temperature=.7/top_p=.95/seed42。最多64个主请求＋64个分支槽位。同步是本实现的限制，不声称复现原256路异步吞吐，不直接拿历史378.6/179.5秒作工程加速基准。异步移植需要另一批实现与验证，当前入口拒绝开启。

## 预算与统计

正式每题共同预算16000：主路输出＋全部试答输出＋强制试答提示token。原题前缀回放不重复算成“生成token”，但单列回放输入token，全部前向、暂停、判定与分支清理时间计入纯生成。最多256个试答输出token不是零额外成本，尚需提示与长前缀回放。没有省去这些成本的加速承诺。

`result.json`区分平均主思考/主输出、全部分支输出、含试答提示的预算token；报告触顶、逐题token IDs、mask位置、每次试答熵/概率/前缀哈希、回放输入与输出token。`generation_seconds`含分支和判定，不含模型加载、作者判分及实测文件写入。`probe_wall_seconds`与`callback_host_seconds`有重叠，不能相加或当净内核成本。工程单token审计另记，不混入方法收益。

## 首批工程协议（尚未授权或冻结题单）

仅1.5B，8道经执行对话对账的训练工程题，R、Roff、Rshadow、C、RC五组，每份主输出上限256，共40份主输出。工程专用间隔32、最多2次、每次8个试答token，Rshadow加每检查点1个回放审计token，最多384个试答输出＋16个审计输出；这些工程配置不得用作正式效果参数。Rshadow工程不从主输出上限扣探测token，以便做等长行为检查，全部开销仍记录，工程不判方法效果。

预计2—6分钟只是预算估计，尚无实测；每组180秒硬停止，外层命令20分钟硬停止，模型加载和原生CPU检查也在外层时间内。失败不重试、不继续筛选。正式筛选另批64道新训练题×R/C/RC，每组600秒，预估5—15分钟不保证；先验正确率损失上限2个百分点，组合应相对R、C都减少包含试答的总输出，触顶不增。小样本点值通过不证明非劣，不自动扩测，缺少U不能声称超加性协同。进一步确认需要新题登记与另批协议。

`coordination.template.json`需要执行对话填写本批授权、题单SHA256和两条线对账结果；`data_usage.json`目前已用/预留均为空。每行题单必须有`train_index`、`problem`、`answer`、`problem_sha256`、`split:"train"`。归一化NFKC后删除Unicode空白。不得把A开发题登记成B确认题。

在隔离部署的仓库根目录，用已有推理环境执行（路径由执行对话按部署根目录填写）：

```bash
P=integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_probe_20260913
PY=/root/autodl-tmp/venvs/easysteer-vllm026/bin/python
"$PY" "$P/prepare.py" --rows /path/to/engineering8.jsonl --coordination /path/to/coordination.json --phase engineering --dataset math --run-id cgrs_probe_engineering8_run1_20260913 --output /path/to/unique_plan.json
timeout 1200s "$PY" "$P/run.py" --plan /path/to/unique_plan.json --output-root /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912
```

沿用原环境PATH中的ninja与CUDA工具，不安装升级环境。输出目录已存在即拒绝。运行前执行对话检查GPU独占、源码/资产/题单身份；本地完成实现不表示该批已授权。筛选还强制要求同源码/资产的成功`engineering_gate.json`及其SHA256。判分入口使用`/root/autodl-tmp/venvs/rebalance/bin/python .../grade.py --output /path/to/run`。

## 本地验证边界

`test_cpu.py`验证纯状态机及用NumPy替身验证的接口合同；`test_native.py`在实际推理环境以CPU Torch验证FP32/BF16熵、行隔离及mask写入，模型加载前自动执行。当前本地无Torch，未安装，所以后者和GPU门槛均未运行，不能写成已验证可运行/已有效。详细本地记录见`cpu_checks.json`。共享源码及旧结果未修改。
