"""Grade the three saved arms once using the author's CPU environment; no GPU import."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from mechanism_candidates import ROOT, BASE, require, read, save, sha
from run_mechanism_screen import validate_bundle,validate_arm


def compare(a,b,ga,gb,indices):
    groups={}
    for label,group,grades in [('original_dynamic',a,ga),('candidate',b,gb)]:
        r=group['records'];n=len(r)
        require(n==len(grades['records'])==100,'Missing scored records')
        groups[label]=dict(count=n,correct=grades['correct'],accuracy_percent=100*grades['correct']/n,
            mean_thinking_tokens=sum(x['thinking_tokens'] for x in r)/n,
            mean_total_tokens=sum(x['tokens'] for x in r)/n,
            capped=sum(x['finish_reason']=='length' or x['tokens']==16000 for x in r),
            generation_seconds=group['summary']['generation_seconds'],
            grading_seconds=grades['seconds'],math_verify_grading_errors=group['summary']['grading_errors'],
            preemptions=group['summary']['preemptions'],dynamic_kv_replay=group['summary']['dynamic_kv_replay'])
    original,candidate=groups.values()
    thinking=100*(candidate['mean_thinking_tokens']/original['mean_thinking_tokens']-1)
    total=100*(candidate['mean_total_tokens']/original['mean_total_tokens']-1)
    accuracy=candidate['accuracy_percent']-original['accuracy_percent']
    per_question=[]
    for i,(x,y,gx,gy) in enumerate(zip(a['records'],b['records'],ga['records'],gb['records'],strict=True)):
        per_question.append(dict(index=i,train_index=indices[i],original_correct=gx['correct'],candidate_correct=gy['correct'],
            thinking_token_delta=y['thinking_tokens']-x['thinking_tokens'],total_token_delta=y['tokens']-x['tokens'],
            original_capped=x['finish_reason']=='length' or x['tokens']==16000,candidate_capped=y['finish_reason']=='length' or y['tokens']==16000))
    passes=thinking<=-5 and total<=-5 and accuracy>=0 and candidate['capped']<=original['capped']
    return dict(groups=groups,thinking_token_change_percent=thinking,total_token_change_percent=total,
        accuracy_change_percentage_points=accuracy,
        improved_indices=[r['index'] for r in per_question if not r['original_correct'] and r['candidate_correct']],
        degraded_indices=[r['index'] for r in per_question if r['original_correct'] and not r['candidate_correct']],
        per_question=per_question,passes_fixed_screen=passes,
        decision='Eligible for separately authorized independent confirmation; accuracy preservation unproven' if passes else 'Stop without retuning or expansion on this set')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analyze-only',action='store_true',help='Use saved author grades locally, never regrade')
    args=parser.parse_args();bundle=args.bundle.resolve();output=args.output.resolve()
    plan,rows=validate_bundle(bundle)
    ledger=read(output/'run_ledger.json')
    require(ledger['status']=='generation_completed_grading_pending','Generation incomplete or already finalized')
    require(ledger['plan_sha256']==sha(bundle/'plan.json'),'Plan mismatch')
    raw={name:read(output/(name+'.json')) for name in plan['run_order']}
    groups={name:validate_arm(raw[name],plan,rows,name) for name in raw}
    for name in raw:require(sha(output/(name+'.json'))==ledger['arms'][name]['raw_sha256'],'Raw output changed')
    require(not (output/'analysis.json').exists(),'Analysis exists; inspect it instead')
    if not args.analyze_only:
        sys.path.insert(0,str(ROOT/'sources/ReBalance'))
        from utils.parser import extract_answer,parse_ground_truth
        from utils.grader import check_is_correct
        for name,group in groups.items():
            target=output/(name+'.author.json')
            require(not target.exists(),'Existing grading must be verified/reused explicitly; no automatic regrade')
            started=time.perf_counter();records=[]
            for i,(row,record) in enumerate(zip(rows,group['records'],strict=True)):
                _,gold=parse_ground_truth(row,'math')
                correct=bool(check_is_correct(extract_answer(record['text'],'math'),gold))
                records.append(dict(index=i,train_index=row['train_index'],correct=correct))
                save(output/(name+'.author.partial.json'),dict(status='partial',records=records,input_sha256=sha(output/(name+'.json'))))
            save(target,dict(status='completed',input_sha256=sha(output/(name+'.json')),dataset_sha256=plan['dataset_sha256'],
                correct=sum(r['correct'] for r in records),seconds=time.perf_counter()-started,records=records,
                grader_sources={f:sha(ROOT/'sources/ReBalance/utils'/f,source=True) for f in ('grader.py','parser.py')}))
            print(name,sum(r['correct'] for r in records),'/100',flush=True)
    grades={name:read(output/(name+'.author.json')) for name in raw}
    for name,grade in grades.items():
        require(grade['status']=='completed' and grade['input_sha256']==sha(output/(name+'.json')),'Grade mismatch')
        require(grade['dataset_sha256']==plan['dataset_sha256'],'Grading dataset mismatch')
        require([r['train_index'] for r in grade['records']]==plan['train_indices'],'Grading pairing mismatch')
        for f,digest in grade['grader_sources'].items():require(sha(ROOT/'sources/ReBalance/utils'/f,source=True)==digest,'Author grader changed')
    result={name:compare(groups['original_dynamic'],groups[name],grades['original_dynamic'],grades[name],plan['train_indices']) for name in plan['run_order'][1:]}
    passing=[name for name in result if result[name]['passes_fixed_screen']]
    chosen=min(passing,key=lambda name:(result[name]['groups']['candidate']['mean_thinking_tokens'],result[name]['groups']['candidate']['mean_total_tokens'],plan['run_order'].index(name))) if passing else None
    save(output/'analysis.json',dict(status='completed',scope=plan['scope'],comparisons=result,
        candidate_for_separate_confirmation=chosen,confirmation_authorized=False,
        source_sha256={name:sha(output/name) for name in [*(n+'.json' for n in raw),*(n+'.author.json' for n in raw),'run_ledger.json']},
        new_generations_in_this_analysis=0))
    print(json.dumps({name:{k:v for k,v in r.items() if k!='per_question'} for name,r in result.items()},ensure_ascii=False))


if __name__=='__main__':main()
