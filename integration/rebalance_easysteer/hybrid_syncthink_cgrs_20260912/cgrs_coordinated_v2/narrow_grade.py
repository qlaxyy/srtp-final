"""Author grading and fixed descriptive screening criteria, not test-set tuning."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from engineering import ROOT,read,save,sha


def decision(groups, primary='RC8'):
    r,c,n=(groups[k] for k in ('R','RC14',primary))
    keys=('mean_total_tokens','mean_thinking_tokens')
    accuracy=n['correct']>=c['correct'] and 100*(n['correct']-r['correct'])/r['n']>=-2
    compression=all(n[k]<r[k] for k in keys)
    caps=n['capped']<=min(r['capped'],c['capped'])
    improvement=n['correct']>c['correct'] or all(n[k]<c[k] for k in keys)
    return dict(accuracy_point_pass=accuracy,compression_vs_R=compression,caps_pass=caps,
                strict_improvement_over_RC14=improvement,promote=accuracy and compression and caps and improvement)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args();folder=args.output
    plan=read(folder/'resolved_plan.json');status=read(folder/'batch_status.json')
    assert status['passed'] and status['phase']=='screen'
    strength=plan.get('candidate_kind')=='strength_screen'
    if strength:
        from strength_screen import validate
        validate(plan)
        assert status['plan_sha256']==sha(folder/'resolved_plan.json')
    assert not (folder/'analysis.json').exists()
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n,True)==h,n
    import sys
    sys.path.insert(0,str(ROOT/'sources/ReBalance'))
    from utils.parser import extract_answer,parse_ground_truth
    from utils.grader import check_is_correct
    from full_grade import compare
    began=time.perf_counter();groups={};raw={};reused={}
    for name in plan['arms']:
        f=folder/name;result=read(f/'result.json');records=result['records']
        assert result['status']=='complete' and len(records)==100
        partial=f/'author_partial.jsonl'
        previous=[json.loads(s) for s in partial.read_text(encoding='utf8').splitlines()] if args.resume and partial.exists() else []
        assert len(previous)<=100;reused[name]=len(previous);labels=[]
        with partial.open('a' if args.resume else 'x',encoding='utf8') as stream:
            for i,(row,rec) in enumerate(zip(plan['rows'],records)):
                assert row['problem_sha256']==rec['problem_sha256'] and rec['dataset_index']==i
                if strength:assert row['problem']==rec['problem'] and row['answer']==rec['answer']
                assert len(rec['token_ids'])==rec['tokens']<=16000
                ids=rec['token_ids']
                assert rec['thinking_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
                assert rec['finish_reason'] in ('stop','length')
                identity=dict(dataset_index=i,train_index=row['train_index'],problem_sha256=row['problem_sha256'],
                              text_sha256=hashlib.sha256(rec['text'].encode()).hexdigest())
                if i<len(previous):
                    label=previous[i];assert all(label[k]==v for k,v in identity.items())
                    assert type(label['correct']) is bool
                else:
                    _,gold=parse_ground_truth(row,'math');answer=extract_answer(rec['text'],'math') if strength else extract_answer(rec['text'])
                    label=dict(identity,correct=bool(check_is_correct(answer,gold)),extracted_answer=answer)
                    stream.write(json.dumps(label,ensure_ascii=False)+'\n');stream.flush()
                labels.append(label)
        values=[float(x[1]) for x in csv.reader((f/'gpu.csv').read_text().splitlines()) if len(x)==4]
        groups[name]=dict(n=100,correct=sum(l['correct'] for l in labels),
            mean_total_tokens=sum(r['tokens'] for r in records)/100,
            mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/100,
            capped=sum(r['finish_reason']=='length' for r in records),
            generation_seconds=result['generation_seconds'],gpu_mean_percent=sum(values)/len(values) if values else None,
            output_tokens_per_second=sum(r['tokens'] for r in records)/result['generation_seconds'],
            replayed_input_tokens=sum(e.get('replay_prefill_tokens',0) for e in result['replay_events']),
            eligible=sum(e['eligible'] for e in result['events'].values()),
            interventions=sum(e['changed'] for e in result['events'].values()),
            scheduler_preemptions=result['scheduler_preemptions'],replay_counts=result['replay_counts'],
            control_gpu_seconds=None,extra_probe_forwards=0,setup_seconds=result['setup_seconds'],
            checkpoint_io_seconds=result['checkpoint_io_seconds'],labels=labels,result_sha256=sha(f/'result.json'))
        raw[name]=[dict(r,correct=l['correct']) for r,l in zip(records,labels)]
        if strength:
            groups[name]['empty_answer_extractions']=sum(not l.get('extracted_answer') for l in labels)
    comparisons={}
    primary=plan.get('primary_candidate','RC8')
    if strength:
        from strength_statistics import comparisons as four_arm_comparisons
        from strength_screen import decision as screen_decision
        comparisons=four_arm_comparisons(raw)
        selected_decision=screen_decision(groups)
    else:
        assert plan['arms']==['R','RC14',primary] and primary in ('RC8','RChistory')
        for base,candidate in [('R','RC14'),('R',primary),('RC14',primary)]:
            comparisons[candidate+'_vs_'+base]=compare(raw[base],raw[candidate],groups[candidate]['labels'])
        selected_decision=decision(groups,primary)
    save(folder/'analysis.json',dict(status='complete_training_screen',groups=groups,comparisons=comparisons,
        decision=selected_decision,reused_labels=reused,grade_seconds=time.perf_counter()-began,
        plan_sha256=sha(folder/'resolved_plan.json'),
        statistics=dict(bootstrap_draws=20000,bootstrap_seed=20260917,shared_question_indices_across_arms=True) if strength else None,
        limitations=['100 paired training questions; not independent confirmation or test evidence',
            'Single seed and fixed arm order; confidence intervals omit batch/sampling variability',
            'Point loss bound and confidence-interval noninferiority are separate',
            'No automatic confirmation, threshold retuning, or runtime selection']))


if __name__=='__main__':main()
