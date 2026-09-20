"""Analyze the explicitly authorized fresh ReBalance speed run, CPU only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare_execution import phash
BASE=ROOT/'.codex_work/rebalance_speed1_verified'
RUN='rebalance_full_speed_c256_run1_20260913'
OUT=BASE/'results/easysteer/hybrid_syncthink_cgrs_20260912'/RUN
PROJECT=BASE/'projects/hybrid_syncthink_cgrs_20260913_speed1'
DEST=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE/'speed_recheck_20260913/results'
DEST.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
resolved=read(OUT/'resolved_plan.json');history=read(PROJECT/'historical_reference.json')
assert sha(PROJECT/'historical_reference.json')==resolved['historical_reference_sha256']
identity=read(OUT/'historical_token_identity.json');status=read(OUT/'batch_status.json')
assert status['status']=='complete'
verified=0;generated=[]
for name,h in resolved['source_sha256'].items():
    p=ROOT/name
    if name=='baseline_model_runner.py':p=ROOT/'.codex_work/baseline_model_runner.py'
    if name.endswith('/_version.py') and not p.exists():generated.append(name);continue
    b=p.read_bytes();assert h in (hashlib.sha256(b).hexdigest(),hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()),name
    verified+=1
results={};pairs=[]
for role,n in [('math_test',500),('gsm8k_test',1319)]:
    raw=read(OUT/role/'R/result.json');grade=read(OUT/role/'R/author_grade.json')
    assert sha(OUT/role/'R/result.json')==grade['input_sha256']
    assert len(raw['records'])==len(grade['records'])==n
    assert raw['control']['statistics_and_mask_gpu_ms']==0 and not raw['control']['requests']
    rows=[json.loads(s) for s in (HERE/'full_tests_20260913'/f'{role}.jsonl').read_text(encoding='utf-8').splitlines()]
    combo_root=ROOT/'.codex_work/hybrid_full1_verified/results/easysteer/hybrid_syncthink_cgrs_20260912/s64_full_math500_gsm1319_RS_c256_run1_20260913'/role/'RS'
    combo=read(combo_root/'result.json');combo_grade=read(combo_root/'author_grade.json')
    assert sha(combo_root/'result.json')==combo_grade['input_sha256']
    for i,(r,g,row,h) in enumerate(zip(raw['records'],grade['records'],rows,identity[role]['records'])):
        assert r['train_index']==g['train_index']==h['dataset_index']==i
        assert r['problem']==row['problem'] and str(r['gold'])==str(row['answer'])
        assert r['prompt_token_ids']==resolved['prompts'][role][i]
        assert len(r['token_ids'])==r['tokens']==h['new_tokens'] and r['tokens']<=16000
        think=r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids'])
        assert r['thinking_tokens']==think and r['hybrid'] is None
        assert hashlib.sha256(json.dumps(r['token_ids'],separators=(',',':')).encode()).hexdigest()==h['new_token_sha256']
        assert history[role]['groups']['R']['records'][i]['problem_sha256']==phash(r['problem'])
        pairs.append(dict(dataset=role,dataset_index=i,problem_sha256=phash(r['problem']),new_correct=g['author_correct'],old_correct=history[role]['groups']['R']['records'][i]['correct'],new_tokens=r['tokens'],new_thinking_tokens=think,combo_correct=combo_grade['records'][i]['author_correct'],combo_tokens=combo['records'][i]['tokens'],**{k:v for k,v in h.items() if k not in ('dataset_index','new_tokens')}))
    old=history[role]['groups']['R']['summary']
    seconds=raw['generation_seconds'];total=sum(r['tokens'] for r in raw['records'])
    new=dict(correct=grade['correct'],accuracy_percent=100*grade['correct']/n,mean_tokens=total/n,mean_thinking_tokens=sum(r['thinking_tokens'] for r in raw['records'])/n,capped=sum(r['tokens']==16000 for r in raw['records']),generation_seconds=seconds,tokens_per_second=total/seconds,checkpoint_io_seconds=raw['checkpoint_io_seconds'],author_grade_seconds=grade['seconds'])
    same=identity[role]['identical'];assert same==sum(r['identical'] for r in identity[role]['records'])
    sample=np.random.default_rng(20260912).integers(0,n,(10000,n))
    ci=lambda a:np.quantile(a,[.025,.975]).tolist()
    comparisons={}
    for name,records,labels in [('historical_R',history[role]['groups']['R']['records'],[r['correct'] for r in history[role]['groups']['R']['records']]),('recent_RS',combo['records'],[r['author_correct'] for r in combo_grade['records']])]:
        a=np.array([r['tokens'] for r in raw['records']]);b=np.array([r['tokens'] for r in records]);x=np.array([r['author_correct'] for r in grade['records']]);y=np.array(labels)
        delta=y.astype(int)-x.astype(int)
        comparisons[name+'_minus_new_R']=dict(total_token_percent_change=100*(b.mean()/a.mean()-1),total_token_percent_change_ci95=ci(100*(b[sample].mean(1)/a[sample].mean(1)-1)),accuracy_delta_pp=100*float(delta.mean()),accuracy_delta_pp_ci95=ci(100*delta[sample].mean(1)),wrong_to_right=np.flatnonzero(~x&y).tolist(),right_to_wrong=np.flatnonzero(x&~y).tolist())
    result=dict(dataset=role,n=n,historical_R=old,current_R=new,recent_RS=dict(correct=combo_grade['correct'],accuracy_percent=100*combo_grade['correct']/n,mean_tokens=sum(r['tokens'] for r in combo['records'])/n,generation_seconds=combo['generation_seconds']),identical_token_sequences=same,changed_token_sequences=n-same,speed=dict(current_R_time_change_percent=100*(seconds/old['generation_seconds']-1),current_R_throughput_change_percent=100*((total/seconds)/(old['mean_tokens']*n/old['generation_seconds'])-1),recent_RS_time_change_vs_current_R_percent=100*(combo['generation_seconds']/seconds-1)),comparisons=comparisons,source_sha256=dict(current_output=sha(OUT/role/'R/result.json'),current_grading=sha(OUT/role/'R/author_grade.json'),combo_output=sha(combo_root/'result.json'),combo_grading=sha(combo_root/'author_grade.json'),frozen_output=identity[role]['frozen_source_sha256']),limitation='One timing sample per condition, not a repeated hardware benchmark. New R has no additional8-question warmup; first-use kernel cost stays in measured time. Cold model load and author grading excluded.')
    results[role]=result;save(DEST/(role+'_analysis.json'),result)
save(DEST/'per_question.json',pairs)
save(DEST/'verification.json',dict(runtime_commit=resolved['deployment']['commit'],source_files_verified=verified,generated_version_not_locally_compared=generated,formal_records_verified=len(pairs),model_and_calibration_assets=resolved['assets'],environment=resolved['environment'],batch_status=status,engineering_reuse=resolved['completed_engineering_reuse'],new_engineering_answers=0,S64_active_requests=0,S64_entropy_or_mask_calls=0,resolved_sha256=sha(OUT/'resolved_plan.json'),historical_token_identity_sha256=sha(OUT/'historical_token_identity.json'),archive=read(ROOT/'.codex_work/speed_archive_receipt.json')))
lines=['# ReBalance全量速度重测','',f'用户明确授权仅重测ReBalance，MATH500＋GSM8K1319，各一次；运行提交`{resolved["deployment"]["commit"]}`。冻结原文件和标签不动。','',
'256路异步、每轮32768 token、BF16、4090D单卡、显存0.9、上下文32768、in-graph，关闭chunked prefill/prefix caching；temperature0.7/top_p0.95/seed42/max_new_tokens16000。原自提向量/fit不变，S64完全关闭，统计和mask开销0；已有相同生产源码的256路工程检查复用，不增加工程答案。','',
'| 数据集 | 组别 | 正确率 | 平均思考token | 平均总token | 触顶 | 纯生成秒 | token/秒 |','|---|---|---:|---:|---:|---:|---:|---:|']
for role,r in results.items():
    for name,a in [('历史ReBalance',r['historical_R']),('本次ReBalance',r['current_R'])]:
        tps=a.get('tokens_per_second',a['mean_tokens']*r['n']/a['generation_seconds'])
        lines.append(f'| {role} | {name} | {a["accuracy_percent"]:.2f}% | {a["mean_thinking_tokens"]:.3f} | {a["mean_tokens"]:.3f} | {a["capped"]} | {a["generation_seconds"]:.2f} | {tps:.1f} |')
lines+=['','## 输出一致性与速度解释','']
for role,r in results.items():
    s=r['speed'];lines+=[f'- {role}：{r["identical_token_sequences"]}/{r["n"]}条token序列与冻结R逐token相同，{r["changed_token_sequences"]}条发生变化；时间{ s["current_R_time_change_percent"]:+.2f}%，吞吐{s["current_R_throughput_change_percent"]:+.2f}%。最近组合相对本次R的时间{s["recent_RS_time_change_vs_current_R_percent"]:+.2f}%。']
    c=r['comparisons']['recent_RS_minus_new_R'];lines+=[f'  最近组合相对本次R：总token {c["total_token_percent_change"]:+.2f}%（95%区间{c["total_token_percent_change_ci95"][0]:+.2f}%至{c["total_token_percent_change_ci95"][1]:+.2f}%），准确率{c["accuracy_delta_pp"]:+.2f}个百分点（95%区间{c["accuracy_delta_pp_ci95"][0]:+.2f}至{c["accuracy_delta_pp_ci95"][1]:+.2f}）。']
lines+=['',f'整批墙钟{status["wall_seconds"]:.2f}秒，新增1819份完整R答案。纯生成扣除实测落盘，不含模型加载/图编译/作者判分；首次采样内核成本保留。不是重复多次硬件基准，不承诺未来每次秒数一致。不对旧结果或参数作事后调整；所有错误、触顶保留。逐题token SHA256、首次分歧、标签和对照区间在results。','',f'原始归档`.codex_work/rebalance_speed1_complete_20260913.tar.gz`，SHA256 `{read(ROOT/".codex_work/speed_archive_receipt.json")["sha256"]}`；新目录和源码独立，A线与共享环境未修改。']
with (DEST.parent/'速度重测结果.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({role:{k:r[k] for k in ('current_R','speed','identical_token_sequences','changed_token_sequences')} for role,r in results.items()},ensure_ascii=False,indent=2))
