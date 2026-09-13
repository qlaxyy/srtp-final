"""Reverify a downloaded mix05 run using only CPU; never regenerate answers."""
import argparse,hashlib,json,pathlib,subprocess,unicodedata
root=pathlib.Path.cwd();ns=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912';here=ns/'mix05_confirm200_20260913'
base=root/'.codex_work/hybrid_mix05_confirm200_run1_verified'
rundir=base/'results/easysteer/hybrid_syncthink_cgrs_20260912/s64_mix05_confirm_math200_gsm200_c256_run1_20260913'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=read(rundir/'resolved_plan.json');assert read(rundir/'batch_status.json')['status']=='complete'
commit=r['deployment']['commit'];verified=0;generated=[]
for name,h in r['source_sha256'].items():
 p=root/name
 if p.exists() and (sha(p)==h or hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h):verified+=1;continue
 if name=='baseline_model_runner.py':
  assert h=='88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5';verified+=1;continue
 if name.endswith('/vllm/_version.py'):generated.append(dict(path=name,remote_sha256=h,reason='runtime-generated version file, recorded not compared with source tree'));continue
 data=subprocess.check_output(['git','show',commit+':'+name])
 assert hashlib.sha256(data).hexdigest()==h,name
 verified+=1
items=[];raw_hashes={};group_count={}
for role in r['expansion']['roles']:
 rows=r['soft2_rows'][role]
 for arm in r['expansion']['arms']:
  p=rundir/role/arm/'result.json';g=p.with_name('author_grade.json');v=read(p);grade=read(g)
  assert grade['input_sha256']==sha(p) and len(v['records'])==len(grade['records'])==len(rows)==200
  for name,h in grade['grader_sha256'].items():
   assert h == r['source_sha256']['sources/ReBalance/utils/'+name]
  raw_hashes[str(p.relative_to(rundir))]=sha(p);raw_hashes[str(g.relative_to(rundir))]=sha(g)
  group_count[role+'/'+arm]=len(rows)
  for source,x,label,prompt in zip(rows,v['records'],grade['records'],r['prompts'][role]):
   assert x['train_index']==source['train_index']==label['train_index']
   assert x['problem']==source['problem'] and x['gold']==source['answer']
   assert x['prompt_token_ids']==prompt
   h=hashlib.sha256(''.join(unicodedata.normalize('NFKC',x['problem']).split()).encode()).hexdigest();assert h==source['problem_sha256']
   ids=x['token_ids'];assert len(ids)==x['tokens']<=16000
   end=ids.index(151649) if 151649 in ids else len(ids)
   assert end==x['thinking_tokens'];assert len(ids)-end-int(151649 in ids)==x['answer_tokens']
   if arm=='R':assert x['hybrid'] is None
   else:
    ctl=x['hybrid'];assert ctl['mode']=='mix05' and ctl['accepted_tokens']==len(ids)
    assert ctl['end_position']==(end if end<len(ids) else -1)
    assert 0<=ctl['discarded_async_tail_tokens']<=1
    assert ctl['first_bias']==-1 or 0<=ctl['first_bias']<len(ids)
   items.append(dict(dataset=source['dataset'],role=role,arm=arm,train_index=x['train_index'],problem_sha256=h,author_correct=label['author_correct'],tokens=x['tokens'],thinking_tokens=x['thinking_tokens'],answer_tokens=x['answer_tokens'],capped=x['tokens']==16000,token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),hybrid=x['hybrid']))
eng={name:read(rundir/'engineering'/name/'result.json')['records'] for name in ('pre_R','off_R','shadow_R','S','RS')}
assert all(len(v)==8 for v in eng.values())
for name in ('off_R','shadow_R'):
 for x,y in zip(eng['pre_R'],eng[name]):
  assert x['token_ids']==y['token_ids'];assert x['R_history']['sha256']==y['R_history']['sha256']
for x,y in zip(eng['pre_R'],eng['RS']):
 i=y['hybrid']['first_bias'];length=i if i>=0 else min(len(x['token_ids']),len(y['token_ids']))
 assert x['token_ids'][:length]==y['token_ids'][:length]
closure=read(rundir/'closure.json');assert not closure['compute_apps'].strip() and not closure['shared_status'].strip();assert closure['shared_commit'].strip()=='82c570e5e3d49971605268c30a59fe856d5d1f01'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
out=args.output;out.mkdir(exist_ok=False)
for name,obj in [('per_question.json',items),('verification.json',dict(runtime_commit=commit,resolved_sha256=sha(rundir/'resolved_plan.json'),source_files_verified=verified,generated_version_files=generated,formal_answers=len(items),new_engineering_answers=0,reused_engineering_answers=40,group_count=group_count,engineering_gate=read(rundir/'engineering_gate.json'),archive=read(root/'.codex_work/mix05_confirm200_archive_receipt.json'),batch_status=read(rundir/'batch_status.json'),closure=closure,raw_sha256=raw_hashes,assets=r['assets'],environment=r['environment'],coordination=r['coordination_receipt'],limitations=['No extra model/logit replay; observed token records only','Independent confirmation on200 new questions per dataset; single seed and batch order','Source generated _version.py recorded separately from Git identity']))]:
 with (out/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
print('VERIFIED',verified,'sources',len(items),'formal answers')
