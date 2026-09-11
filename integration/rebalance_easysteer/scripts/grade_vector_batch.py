"""Author-grade a saved vector pair once; errors and caps remain in all metrics."""
import argparse
import json
from pathlib import Path
import sys
import time
from mechanism_candidates import ROOT, require, read, save, sha
from run_vector_batch import validate_bundle,validate_arm


def comparison(groups,grades,indices):
    names=list(groups);require(len(names)==2,'Paired comparison only')
    stats={};n=len(indices)
    for name in names:
        group=groups[name];records=group['records'];grade=grades[name]
        require(len(records)==len(grade['records'])==n,'Missing records')
        require(grade['correct']==sum(r['correct'] for r in grade['records']),'Incorrect grade total')
        stats[name]=dict(count=n,correct=grade['correct'],accuracy_percent=100*grade['correct']/n,
            mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/n,
            mean_total_tokens=sum(r['tokens'] for r in records)/n,
            capped=sum(r['finish_reason']=='length' or r['tokens']==16000 for r in records),
            generation_seconds=group['summary']['generation_seconds'],grading_seconds=grade['seconds'],
            preemptions=group['summary']['preemptions'],dynamic_kv_replay=group['summary']['dynamic_kv_replay'])
    first,second=names;a,b=stats.values()
    thinking=100*(b['mean_thinking_tokens']/a['mean_thinking_tokens']-1)
    total=100*(b['mean_total_tokens']/a['mean_total_tokens']-1)
    per=[]
    for i,(x,y,gx,gy) in enumerate(zip(groups[first]['records'],groups[second]['records'],grades[first]['records'],grades[second]['records'],strict=True)):
        per.append(dict(index=i,train_index=indices[i],original_correct=gx['correct'],candidate_correct=gy['correct'],
            thinking_token_delta=y['thinking_tokens']-x['thinking_tokens'],total_token_delta=y['tokens']-x['tokens'],
            original_capped=x['finish_reason']=='length' or x['tokens']==16000,
            candidate_capped=y['finish_reason']=='length' or y['tokens']==16000))
    passed=thinking<=-5 and total<=-5 and b['correct']>=a['correct'] and b['capped']<=a['capped']
    return dict(groups=stats,thinking_token_change_percent=thinking,total_token_change_percent=total,
        accuracy_change_percentage_points=b['accuracy_percent']-a['accuracy_percent'],
        improved_indices=[r['index'] for r in per if not r['original_correct'] and r['candidate_correct']],
        degraded_indices=[r['index'] for r in per if r['original_correct'] and not r['candidate_correct']],
        per_question=per,passes_fixed_gate=passed)


def controlled_comparison(groups,grades,indices):
    candidate='latent_feedback_clip'
    comparisons={name:comparison({n:groups[n] for n in [name,candidate]},
        {n:grades[n] for n in [name,candidate]},indices)
        for name in ['original_dynamic','feedback_disabled']}
    return dict(comparisons=comparisons,passes_fixed_gate=all(v['passes_fixed_gate'] for v in comparisons.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    a=p.parse_args();bundle=a.bundle.resolve();out=a.output.resolve();plan,rows=validate_bundle(bundle)
    ledger=read(out/'run_ledger.json')
    require(ledger['status']=='generation_completed_grading_pending' and ledger['plan_sha256']==sha(bundle/'plan.json'),'Incomplete/mismatched generation')
    require(not (out/'analysis.json').exists(),'Analysis already exists')
    raw={n:read(out/(n+'.json')) for n in plan['run_order']}
    groups={n:validate_arm(r,plan,rows,n) for n,r in raw.items()}
    digests={n:sha(out/(n+'.json')) for n in raw}
    for name,digest in digests.items():require(digest==ledger['arms'][name]['raw_sha256'],'Raw changed')
    grader_sha={n:sha(ROOT/'sources/ReBalance/utils'/n,source=True) for n in ('grader.py','parser.py')}
    if not a.analyze_only:
        sys.path.insert(0,str(ROOT/'sources/ReBalance'))
        from utils.parser import extract_answer,parse_ground_truth
        from utils.grader import check_is_correct
        for name,group in groups.items():
            target=out/(name+'.author.json');require(not target.exists(),'Grade exists; verify/reuse explicitly')
            started=time.perf_counter();scored=[]
            with (out/(name+'.author.partial.jsonl')).open('x',encoding='utf-8') as stream:
                for i,(row,record) in enumerate(zip(rows,group['records'],strict=True)):
                    _,gold=parse_ground_truth(row,'math')
                    correct=bool(check_is_correct(extract_answer(record['text'],'math'),gold))
                    item=dict(index=i,train_index=row['train_index'],correct=correct)
                    scored.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
            save(target,dict(status='completed',input_sha256=digests[name],dataset_sha256=plan['dataset_sha256'],
                correct=sum(r['correct'] for r in scored),seconds=time.perf_counter()-started,records=scored,grader_sources=grader_sha))
            print(name,sum(r['correct'] for r in scored),'/',len(rows),flush=True)
    grades={n:read(out/(n+'.author.json')) for n in raw}
    for name,grade in grades.items():
        require(grade['status']=='completed' and grade['input_sha256']==digests[name],'Grade mismatch')
        require(grade['dataset_sha256']==plan['dataset_sha256'] and grade['grader_sources']==grader_sha,'Grader/data changed')
        require([r['train_index'] for r in grade['records']]==plan['train_indices'],'Score pairing changed')
    if plan.get('graph_control_comparison'):
        result=controlled_comparison(groups,grades,plan['train_indices'])
        comparisons=result['comparisons']
    else:
        result=comparison(groups,grades,plan['train_indices'])
    save(out/'analysis.json',dict(status='completed',scope=plan['scope'],stage=plan['stage'],comparison=result,
        candidate=plan['run_order'][-1],eligible_for_confirmation=plan['stage']=='screen' and result['passes_fixed_gate'],
        new_generations_in_analysis=0,source_sha256={n:sha(out/n) for n in [*(x+'.json' for x in raw),*(x+'.author.json' for x in raw),'run_ledger.json']}))
    compact=result if not plan.get('graph_control_comparison') else {name:{k:v for k,v in item.items() if k!='per_question'} for name,item in comparisons.items()}
    print(json.dumps({k:v for k,v in compact.items() if k!='per_question'}))


if __name__=='__main__':main()
