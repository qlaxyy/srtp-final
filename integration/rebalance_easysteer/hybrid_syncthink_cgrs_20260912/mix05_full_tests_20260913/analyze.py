import hashlib,json,sys,subprocess,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[4]
ns=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'
sys.path.insert(0,str(ns))
from prepare_execution import phash
base=root/'.codex_work/hybrid_mix05_full1_verified'
run='mix05_full_math500_gsm1319_RS_c256_run1_20260913'
out=base/'results/easysteer/hybrid_syncthink_cgrs_20260912'/run
project=base/'projects/hybrid_syncthink_cgrs_20260913_mix05_full1'
dest=ns/'mix05_full_tests_20260913/results';dest.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
resolved=read(out/'resolved_plan.json');history=read(project/'historical_mix05_full_reference.json')
assert sha(project/'historical_mix05_full_reference.json')==resolved['historical_reference_sha256']
status=read(out/'batch_status.json');assert status['status']=='complete'
verified=0;generated=[]
for name,h in resolved['source_sha256'].items():
 p=root/name
 if p.exists() and h in [sha(p),hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()]:verified+=1;continue
 if name=='baseline_model_runner.py':
  assert h=='88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5';verified+=1;continue
 if name.endswith('/_version.py'):generated.append(dict(path=name,sha256=h));continue
 data=subprocess.check_output(['git','show',resolved['deployment']['commit']+':'+name])
 assert hashlib.sha256(data).hexdigest()==h,name
 verified+=1
def cp_upper(k,n,alpha=.025):
 if k==n:return 1.
 log_choose=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1) for j in range(k+1)]
 lo,hi=0.,1.
 for _ in range(80):
  p=(lo+hi)/2
  cdf=math.fsum(math.exp(c+j*math.log(p)+(n-j)*math.log1p(-p)) for j,c in enumerate(log_choose))
  if cdf>alpha:lo=p
  else:hi=p
 return (lo+hi)/2

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
  assert h['mode']=='mix05'
  assert h['first_bias']==-1 or 0<=h['first_bias']<r['tokens']
  assert h['end_position']==(count if count<len(r['token_ids']) else -1)
  for k in ('U','R'):assert history[role]['groups'][k]['records'][i]['problem_sha256']==phash(row['problem'])
  paired.append(dict(dataset=role,dataset_index=i,problem_sha256=phash(row['problem']),arms={k:{f:np.asarray(v[f][i]).item() for f in ('tokens','thinking_tokens','correct','capped')} for k,v in data.items()},hybrid=h))
 arms={}
 for k,v in data.items():
  seconds=raw['generation_seconds'] if k=='RS' else history[role]['groups'][k]['summary']['generation_seconds']
  arms[k]=dict(correct=int(v['correct'].sum()),accuracy_percent=100*float(v['correct'].mean()),mean_tokens=float(v['tokens'].mean()),mean_thinking_tokens=float(v['thinking_tokens'].mean()),capped=int(v['capped'].sum()),generation_seconds=seconds,tokens_per_second=float(v['tokens'].sum())/seconds,timing_is_historical=k!='RS')
 assert arms['RS']['correct']==grade['correct']
 sample=np.random.default_rng(20260913).integers(0,n,(10000,n))
 ci=lambda x:np.quantile(x,[.025,.975]).tolist()
 comps={}
 for k in ('U','R'):
  c={}
  for field in ('tokens','thinking_tokens'):
   x,y=data[k][field],data['RS'][field]
   c[field]=dict(delta=float((y-x).mean()),delta_ci95=ci((y-x)[sample].mean(1)),percent_change=100*float(y.mean()/x.mean()-1),percent_change_ci95=ci(100*(y[sample].mean(1)/x[sample].mean(1)-1)))
  x,y=data[k]['correct'],data['RS']['correct'];delta=y.astype(int)-x.astype(int)
  c.update(accuracy_delta_pp=100*float(delta.mean()),accuracy_delta_pp_ci95=ci(100*delta[sample].mean(1)),wrong_to_right=np.flatnonzero(~x&y).tolist(),right_to_wrong=np.flatnonzero(x&~y).tolist(),time_percent_change=100*(arms['RS']['generation_seconds']/arms[k]['generation_seconds']-1))
  loss_upper=cp_upper(int((delta==-1).sum()),n)- (1-cp_upper(n-int((delta==1).sum()),n))
  c.update(net_loss_probability_upper95_conservative=loss_upper,accuracy_2pp_noninferiority_supported=loss_upper<=.02)
  comps['RS_minus_'+k]=c
 control=raw['control'];result=dict(dataset=role,n=n,arms=arms,comparisons=comps,control=dict(statistics_mask_gpu_seconds=control['statistics_and_mask_gpu_ms']/1000,extra_method_forwards=control['extra_model_forwards'],probe_tokens=control['probe_tokens'],biased_requests=sum(r['hybrid']['first_bias']>=0 for r in raw['records']),worker_trigger_count=sum(r['hybrid']['trigger_count'] for r in raw['records']),worker_bias_count=sum(r['hybrid']['bias_count'] for r in raw['records']),worker_revived_end_count=sum(r['hybrid']['revived_end_count'] for r in raw['records']),discarded_async_tail_tokens=sum(r['hybrid']['discarded_async_tail_tokens'] for r in raw['records'])),author_grade_seconds=grade['seconds'],checkpoint_io_seconds=raw['checkpoint_io_seconds'],source_sha256=dict(result=sha(out/role/'RS/result.json'),grade=sha(out/role/'RS/author_grade.json'),historical_evaluation=history[role]['evaluation_sha256'],historical_grading=history[role]['grading_sha256']),bootstrap=dict(repetitions=10000,seed=20260913,unit='same-question pairing with historical outputs'),limitations=['One seed, historical comparators; no contemporaneous baseline rerun','No full standalone S: factorial synergy unidentifiable','Previous mix05 training confirmation remained inconclusive; no retuning on full test outcomes','One seed and batch order; bootstrap excludes batch coupling and timing variability'])
 c=comps['RS_minus_R']
 result['supportive_full_benchmark_criterion']=(c['accuracy_2pp_noninferiority_supported'] and c['tokens']['delta_ci95'][1]<0 and c['thinking_tokens']['delta_ci95'][1]<0 and arms['RS']['capped']<=arms['R']['capped'] and result['control']['biased_requests']>0)
 results[role]=result;save(dest/(role+'_analysis.json'),result)
save(dest/'per_question.json',paired)
save(dest/'verification.json',dict(runtime_commit=resolved['deployment']['commit'],source_files_verified=verified,generated_version_not_locally_compared=generated,formal_records_verified=len(paired),model_and_calibration_assets=resolved['assets'],environment=resolved['environment'],batch_status=status,engineering_gate=read(out/'engineering_gate.json'),resolved_sha256=sha(out/'resolved_plan.json'),historical_reference_sha256=sha(project/'historical_mix05_full_reference.json'),archive=read(root/'.codex_work/mix05_full_archive_receipt.json')))
usage=read(ns/'mix05_full_tests_20260913/data_usage.json')
for r in usage['entries']:r['status']='used_fixed_mix05_full_benchmark';r['run_id']=run
save(dest/'data_usage.json',usage)
print(json.dumps({role:dict(arms=v['arms'],vs_R=v['comparisons']['RS_minus_R'],supportive=v['supportive_full_benchmark_criterion']) for role,v in results.items()},ensure_ascii=True,indent=2))
