import hashlib,json,sys,time,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/projects/hybrid_cgrs_repeatability100x3_20260916')
o=Path('/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/cgrs_history_repeatability100x3_20260916')
sys.path.insert(0,str(r/'sources/ReBalance'))
from utils.parser import extract_answer,parse_ground_truth
from utils.grader import check_is_correct
plan=json.loads((o/'resolved_plan.json').read_text());start=time.time();groups={};partial={}
for name in ('parser.py','grader.py'):
    p=r/'sources/ReBalance/utils'/name;key=p.relative_to(r).as_posix()
    assert hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==plan['source_sha256'][key]
for case in plan['schedule']:
    f=o/case['name'];path=f/'result.json'
    if not path.exists():
        lines=(f/'partial.jsonl').read_text().splitlines() if (f/'partial.jsonl').exists() else []
        partial[case['name']]=dict(completed_outputs=len(lines),status='unfinished_arm_excluded_from_comparison')
        continue
    d=json.loads(path.read_text());assert d['status']=='complete' and len(d['records'])==100
    records=sorted(d['records'],key=lambda x:x['dataset_index']);labels=[]
    with (f/'stopped_batch_author_labels.jsonl').open('x') as stream:
        for i,rec in enumerate(records):
            row=plan['rows'][i];assert rec['dataset_index']==i and rec['problem_sha256']==row['problem_sha256']
            assert len(rec['token_ids'])==rec['tokens']<=16000
            _,gold=parse_ground_truth(row,'math');answer=extract_answer(rec['text'],'math')
            label=dict(dataset_index=i,train_index=row['train_index'],problem_sha256=row['problem_sha256'],correct=bool(check_is_correct(answer,gold)),answer=answer,text_sha256=hashlib.sha256(rec['text'].encode()).hexdigest())
            stream.write(json.dumps(label)+'\n');stream.flush();labels.append(label)
    groups[case['name']]=dict(n=100,correct=sum(x['correct'] for x in labels),mean_total_tokens=sum(x['tokens'] for x in records)/100,mean_thinking_tokens=sum(x['thinking_tokens'] for x in records)/100,capped=sum(x['finish_reason']=='length' for x in records),generation_seconds=d['generation_seconds'],labels=labels,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
comparisons={}
for seed in plan['seeds']:
    if not all(f's{seed}_{a}' in groups for a in plan['arms']):continue
    for a,b in [('R','RC14'),('R','RChistory'),('RC14','RChistory')]:
        x,y=groups[f's{seed}_{a}'],groups[f's{seed}_{b}']
        comparisons[f's{seed}_{b}_vs_{a}']=dict(accuracy_delta_pp=y['correct']-x['correct'],total_change_percent=100*(y['mean_total_tokens']/x['mean_total_tokens']-1),thinking_change_percent=100*(y['mean_thinking_tokens']/x['mean_thinking_tokens']-1),cap_delta=y['capped']-x['capped'],wrong_to_right=[q['train_index'] for p,q in zip(x['labels'],y['labels']) if not p['correct'] and q['correct']],right_to_wrong=[q['train_index'] for p,q in zip(x['labels'],y['labels']) if p['correct'] and not q['correct']])
summary=dict(status='user_stopped_descriptive_only',groups=groups,incomplete=partial,comparisons=comparisons,grade_seconds=time.time()-start,unique_questions=100,preplanned_nine_arm_protocol_completed=False,no_promotion_decision=True,notes='Report each fully completed seed separately. Third seed incomplete, no pooled confirmatory claim, no replacement run.')
(o/'stopped_batch_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(dict(groups={k:{a:b for a,b in v.items() if a!='labels'} for k,v in groups.items()},incomplete=partial,comparisons=comparisons,grade_seconds=summary['grade_seconds'])))
