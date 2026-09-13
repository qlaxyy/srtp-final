import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3]
ns=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'
sys.path.insert(0,str(ns))
from prepare_execution import phash
base=root/'.codex_work/hybrid_full1_verified'
run='s64_full_math500_gsm1319_RS_c256_run1_20260913'
out=base/'results/easysteer/hybrid_syncthink_cgrs_20260912'/run
project=base/'projects/hybrid_syncthink_cgrs_20260913_full1'
dest=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ns/'full_tests_20260913/results';dest.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
resolved=read(out/'resolved_plan.json');history=read(project/'historical_reference.json')
assert sha(project/'historical_reference.json')==resolved['historical_reference_sha256']
status=read(out/'batch_status.json');assert status['status']=='complete'
verified=0;generated=[]
for name,h in resolved['source_sha256'].items():
 p=root/name
 if name=='baseline_model_runner.py':p=root/'.codex_work/baseline_model_runner.py'
 if name.endswith('/_version.py') and not p.exists():generated.append(name);continue
 b=p.read_bytes();assert h in [hashlib.sha256(b).hexdigest(),hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()],name
 verified+=1
results={};paired=[]
for role,n in [('math_test',500),('gsm8k_test',1319)]:
 raw=read(out/role/'RS/result.json');grade=read(out/role/'RS/author_grade.json')
 assert grade['input_sha256']==sha(out/role/'RS/result.json')
 rows=[json.loads(s) for s in (ns/'full_tests_20260913'/f'{role}.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(rows)==len(raw['records'])==len(grade['records'])==n
 data={k:{f:np.array([r[f] for r in g['records']]) for f in ('tokens','thinking_tokens','correct','capped')} for k,g in history[role]['groups'].items()}
 data['RS']={f:np.array([r[f] for r in raw['records']]) for f in ('tokens','thinking_tokens')}
 data['RS']['correct']=np.array([g['author_correct'] for g in grade['records']]);data['RS']['capped']=data['RS']['tokens']==16000
 for i,(row,r,g) in enumerate(zip(rows,raw['records'],grade['records'])):
  assert r['train_index']==g['train_index']==i and row['problem']==r['problem'] and str(row['answer'])==str(r['gold'])
  assert r['prompt_token_ids']==resolved['prompts'][role][i]
  assert len(r['token_ids'])==r['tokens']<=16000
  count=r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids'])
  assert count==r['thinking_tokens'] and r['answer_tokens']==r['tokens']-count-int(151649 in r['token_ids'])
  h=r['hybrid'];assert h['accepted_tokens']==r['tokens'] and h['worker_sampled_tokens']-r['tokens']==h['discarded_async_tail_tokens'] in (0,1)
  if h['first_trigger']>=0:assert r['token_ids'][h['first_trigger']]==151649
  for k in ('U','R'):assert history[role]['groups'][k]['records'][i]['problem_sha256']==phash(row['problem'])
  paired.append(dict(dataset=role,dataset_index=i,problem_sha256=phash(row['problem']),arms={k:{f:np.asarray(v[f][i]).item() for f in ('tokens','thinking_tokens','correct','capped')} for k,v in data.items()},hybrid=h))
 arms={}
 for k,v in data.items():
  seconds=raw['generation_seconds'] if k=='RS' else history[role]['groups'][k]['summary']['generation_seconds']
  arms[k]=dict(correct=int(v['correct'].sum()),accuracy_percent=100*float(v['correct'].mean()),mean_tokens=float(v['tokens'].mean()),mean_thinking_tokens=float(v['thinking_tokens'].mean()),capped=int(v['capped'].sum()),generation_seconds=seconds,tokens_per_second=float(v['tokens'].sum())/seconds,timing_is_historical=k!='RS')
 assert arms['RS']['correct']==grade['correct']
 sample=np.random.default_rng(20260912).integers(0,n,(10000,n))
 ci=lambda x:np.quantile(x,[.025,.975]).tolist()
 comps={}
 for k in ('U','R'):
  c={}
  for field in ('tokens','thinking_tokens'):
   x,y=data[k][field],data['RS'][field]
   c[field]=dict(delta=float((y-x).mean()),delta_ci95=ci((y-x)[sample].mean(1)),percent_change=100*float(y.mean()/x.mean()-1),percent_change_ci95=ci(100*(y[sample].mean(1)/x[sample].mean(1)-1)))
  x,y=data[k]['correct'],data['RS']['correct'];delta=y.astype(int)-x.astype(int)
  c.update(accuracy_delta_pp=100*float(delta.mean()),accuracy_delta_pp_ci95=ci(100*delta[sample].mean(1)),wrong_to_right=np.flatnonzero(~x&y).tolist(),right_to_wrong=np.flatnonzero(x&~y).tolist(),time_percent_change=100*(arms['RS']['generation_seconds']/arms[k]['generation_seconds']-1))
  comps['RS_minus_'+k]=c
 control=raw['control'];result=dict(dataset=role,n=n,arms=arms,comparisons=comps,control=dict(statistics_mask_gpu_seconds=control['statistics_and_mask_gpu_ms']/1000,extra_method_forwards=control['extra_model_forwards'],probe_tokens=control['probe_tokens'],forced=sum(r['hybrid']['first_trigger']>=0 for r in raw['records']),discarded_async_tail_tokens=sum(r['hybrid']['discarded_async_tail_tokens'] for r in raw['records'])),author_grade_seconds=grade['seconds'],checkpoint_io_seconds=raw['checkpoint_io_seconds'],source_sha256=dict(result=sha(out/role/'RS/result.json'),grade=sha(out/role/'RS/author_grade.json'),historical_evaluation=history[role]['evaluation_sha256'],historical_grading=history[role]['grading_sha256']),bootstrap=dict(repetitions=10000,seed=20260912,unit='same-question pairing with historical outputs'),limitations=['One seed, historical comparators; no contemporaneous baseline rerun','No full standalone S: factorial synergy unidentifiable','Failed independent training confirmation remains failed; full tests not used for parameter tuning'])
 results[role]=result;save(dest/(role+'_analysis.json'),result)
save(dest/'per_question.json',paired)
save(dest/'verification.json',dict(runtime_commit=resolved['deployment']['commit'],source_files_verified=verified,generated_version_not_locally_compared=generated,formal_records_verified=len(paired),model_and_calibration_assets=resolved['assets'],environment=resolved['environment'],batch_status=status,engineering_gate=read(out/'engineering_gate.json'),resolved_sha256=sha(out/'resolved_plan.json'),historical_reference_sha256=sha(project/'historical_reference.json'),archive=read(root/'.codex_work/full_archive_receipt.json')))
usage=read(ns/'full_tests_20260913/data_usage.json')
for r in usage['entries']:r['status']='used_fixed_full_test';r['run_id']=run
save(dest/'data_usage.json',usage)
lines=['# 组合全量测试与历史基线对照','',f'用户指定仅测组合；MATH-500与GSM8K 1319题各一次，共1819份新答案。运行提交`{resolved["deployment"]["commit"]}`。旧U/R按冻结文件哈希、作者标签和逐题内容复用，未重跑。','',
'256路并发、异步、每轮32768 token、上下文32768、BF16、单4090D、显存0.9、in-graph；关闭chunked prefill/prefix caching。每题seed42、temperature0.7、top_p0.95、max_new_tokens16000。仍为原ReBalance＋S64（0.8、64），不含CGRS。','',
'| 数据集 | 组别 | 正确率 | 正确数 | 平均思考token | 平均总token | 触顶 | 纯生成秒 |','|---|---|---:|---:|---:|---:|---:|---:|']
for role,r in results.items():
 for k,a in r['arms'].items():lines.append(f'| {role} | {k} | {a["accuracy_percent"]:.2f}% | {a["correct"]}/{r["n"]} | {a["mean_thinking_tokens"]:.3f} | {a["mean_tokens"]:.3f} | {a["capped"]} | {a["generation_seconds"]:.2f} |')
lines+=['','U为历史无干预，R为历史自校准ReBalance，RS为本批组合。历史时间不是同期速度因果对照；两边都包含引擎生成循环，新批扣除实测落盘时间，不计启动和判分。','']
for role,r in results.items():
 lines += [f'## {role}','']
 for k in ('U','R'):
  c=r['comparisons']['RS_minus_'+k];t=c['tokens'];v=c['thinking_tokens']
  lines += [f'- 相对{k}：总token {t["percent_change"]:+.2f}%（95%区间{t["percent_change_ci95"][0]:+.2f}%至{t["percent_change_ci95"][1]:+.2f}%），思考token {v["percent_change"]:+.2f}%；正确率{c["accuracy_delta_pp"]:+.2f}个百分点（95%区间{c["accuracy_delta_pp_ci95"][0]:+.2f}至{c["accuracy_delta_pp_ci95"][1]:+.2f}）。',f'  错→对{len(c["wrong_to_right"])}题：{c["wrong_to_right"]}；对→错{len(c["right_to_wrong"])}题：{c["right_to_wrong"]}。',f'  相对历史生成时间{c["time_percent_change"]:+.2f}%。']
 lines += ['',f'额外方法前向/探测token均0；统计与mask事件区间{r["control"]["statistics_mask_gpu_seconds"]:.2f}秒（包含异步等待，不能当精确净内核成本）。实际触发{r["control"]["forced"]}题；丢弃尾token {r["control"]["discarded_async_tail_tokens"]}个保留在worker账本、不计输出。作者判分{r["author_grade_seconds"]:.2f}秒、落盘{r["checkpoint_io_seconds"]:.3f}秒。','']
lines += ['## 证据边界与资产','',f'整批墙钟{status["wall_seconds"]:.2f}秒，含40份原工程题短输出和1819份正式答案。全部错误、触顶均纳入。结果未用于调参数，也不改写此前MATH200/GSM8K200独立确认失败。未测全量S单项，不能声称同时胜过两个单项或严格协同。配对bootstrap10000次、seed20260912，仅反映本次单seed逐题变化；历史采样与执行源码差异仍是局限。',
'','此前只按03手册的128路配置，未完全对齐378.55/179.48秒的原快跑。此次原日志证实256路，默认参数源码也一致；提示词函数AST与冻结运行提交836e37b相同，模型、向量、fit、采样和判分已核验。默认关闭/只读shadow的异步工程检查通过，实际抢占0；没有实现或测试抢占恢复，遇到抢占即停。',
'',f'原始归档`.codex_work/hybrid_full1_complete_20260913.tar.gz`，SHA256 `{read(root/".codex_work/full_archive_receipt.json")["sha256"]}`。本目录results保存逐题比较、分析和核验。旧基线输出不动、冻结标签不动、共享服务器工作区不动；测试题登记为固定测试用途，不成为后续开发数据。']
with (dest.parent/'全量结果.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({role:dict(arms=r['arms'],vs_R=r['comparisons']['RS_minus_R']) for role,r in results.items()},ensure_ascii=False,indent=2))
