import hashlib,json,sys
from pathlib import Path
raw=Path('E:/srtp/B-history-full1p5b_0915')
repo=Path('E:/srtp/hybrid-syncthink-cgrs-20260912')
v=repo/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
sys.path.insert(0,str(v))
from full_grade import compare
def read(p):return json.loads(p.read_text(encoding='utf8'))
manifest=read(raw/'artifact_manifest.json')
for name,digest in manifest.items():assert hashlib.sha256((raw/name).read_bytes()).hexdigest()==digest,name
analysis=read(raw/'results/analysis.json');plan=read(raw/'results/resolved_plan.json')
assert plan==read(raw/'execution/plan.json')
references=read(raw/'results/historical_reference.json');summary={};per_question={}
for role,sub in plan['datasets'].items():
    result=read(raw/'results'/role/'RChistory/result.json');records=result['records'];s=analysis['datasets'][role]
    assert len(records)==s['n']==len(sub['rows'])
    assert result['status']=='complete'
    labels=s['grades'];events=result['events'];rows=[]
    for i,(row,r,label) in enumerate(zip(sub['rows'],records,labels)):
        assert row['problem_sha256']==r['problem_sha256']==label['problem_sha256']
        assert r['dataset_index']==label['dataset_index']==i
        assert len(r['token_ids'])==r['tokens']<=16000
        assert r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else r['tokens'])
        assert hashlib.sha256(r['text'].encode()).hexdigest()==label['text_sha256']
        event=events[r['request_id']]
        assert event['first_change']<0 or 0<=event['first_reflection']<event['first_change']
        rows.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=label['correct'],
            tokens=r['tokens'],thinking_tokens=r['thinking_tokens'],capped=r['finish_reason']=='length',
            reference={name:data['records'][i] for name,data in references[role]['groups'].items()},events=event))
    assert s['correct']==sum(l['correct'] for l in labels)
    assert s['mean_total_tokens']==sum(r['tokens'] for r in records)/len(records)
    assert s['mean_thinking_tokens']==sum(r['thinking_tokens'] for r in records)/len(records)
    assert s['capped']==sum(r['finish_reason']=='length' for r in records)
    for arm,ref in references[role]['groups'].items():assert compare(ref['records'],records,labels)==s['comparisons'][arm]
    new={k:value for k,value in s.items() if k!='grades'}
    new['reference_summaries']={name:{'n':len(data['records']),'correct':sum(r['correct'] for r in data['records']),
        'mean_total_tokens':sum(r['tokens'] for r in data['records'])/len(data['records']),
        'mean_thinking_tokens':sum(r['thinking_tokens'] for r in data['records'])/len(data['records']),
        'capped':sum(r['capped'] for r in data['records']),
        'historical_generation_seconds':data['summary']['generation_seconds']} for name,data in references[role]['groups'].items()}
    new['scheduler_preemption_count']=len(result['scheduler_preemptions'])
    new['discarded_computed_tokens']=sum(x['computed_tokens_discarded'] for x in result['scheduler_preemptions'])
    new['vs_RC14_time_percent']=100*(result['generation_seconds']/new['reference_summaries']['RC14']['historical_generation_seconds']-1)
    new['vs_RC14_time_note']='Historical pipeline timing, not simultaneous speed benchmark'
    summary[role]=new;per_question[role]=rows
out=v/'first_reflection_full1p5b_20260915/results';out.mkdir(exist_ok=False)
report=dict(status='complete_exposed_full_test_descriptive',datasets=summary,
    total_generation_seconds=sum(s['generation_seconds'] for s in summary.values()),
    process=read(raw/'launch/history_complete.json'),closure=read(raw/'closure.json'),
    code_commit=read(raw/'closure.json')['execution_commit'],verified_files=len(manifest),
    input_plan_sha256=hashlib.sha256((raw/'execution/plan.json').read_bytes()).hexdigest(),
    resolved_plan_sha256=hashlib.sha256((raw/'results/resolved_plan.json').read_bytes()).hexdigest(),
    original_plan_preserved=True,all_metrics_and_bootstraps_reproduced=True,author_labels_reused=True,
    no_probe_forwards=True,confirmation200_used=False,
    limitations=analysis['limitations']+plan['limitations']+[plan['attribution_limit']],raw_root=str(raw))
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(out/'per_question.json').write_text(json.dumps(per_question,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,ensure_ascii=False))
