"""CPU-only reconstruction of possible decode batch-size transitions."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    resolved=read(a.run/'resolved_plan.json')
    assert read(a.run/'batch_status.json')['status']=='complete'
    sizes=[1,2,4,8,16,24,32,40,48,56,64]
    results={}
    inputs={}
    for role in resolved['expansion']['roles']:
        records={}
        for arm in ('R','RS'):
            path=a.run/role/arm/'result.json'
            inputs[str(path.relative_to(a.run))]=hashlib.sha256(path.read_bytes()).hexdigest()
            records[arm]=read(path)['records']
        lengths={arm:[r['tokens'] for r in rows] for arm,rows in records.items()}
        def shape(arm,t,lag):
            n=sum(n>t-lag for n in lengths[arm])
            return dict(active=n,padded=sizes[bisect.bisect_left(sizes,n)] if n else 0)
        transitions=[]
        for lag in (0,1):
            t=next(t for t in range(16000) if shape('R',t,lag)['padded']!=shape('RS',t,lag)['padded'])
            transitions.append(dict(assumed_completion_lag=lag,first_padded_difference=t,
                R=shape('R',t,lag),RS=shape('RS',t,lag)))
        mismatch=[]; earliest=[]
        for x,y in zip(records['R'],records['RS']):
            assert x['train_index']==y['train_index']
            ids,other=x['token_ids'],y['token_ids']
            common=next((i for i,(a,b) in enumerate(zip(ids,other)) if a!=b),min(len(ids),len(other)))
            first=y['hybrid']['first_bias']
            if ids!=other:
                earliest.append(dict(train_index=x['train_index'],position=common,first_bias=first,
                    R_token=ids[common] if common<len(ids) else None,
                    RS_token=other[common] if common<len(other) else None))
            if common<(first if first>=0 else min(len(ids),len(other))):
                mismatch.append(dict(train_index=x['train_index'],first_difference=common,first_bias=first,
                    modeled_R=shape('R',common,0),modeled_RS=shape('RS',common,0),
                    after_both_modeled_transitions=all(common>=r['first_padded_difference'] for r in transitions)))
        results[role]=dict(prompt_tokens=sum(map(len,resolved['prompts'][role])),
            modeled_transitions=transitions,prefix_mismatches=sorted(mismatch,key=lambda x:x['first_difference']),
            mismatch_count=len(mismatch),all_after_modeled_transition=all(x['after_both_modeled_transitions'] for x in mismatch),
            earliest_output_difference=min(earliest,key=lambda x:x['position']))
    result=dict(status='strong_circumstantial_evidence_not_causal_confirmation',datasets=results,input_sha256=inputs,
        assumptions=['All64 requests start one prefill and decode together; prompt sums fit32768, but actual scheduler trace was not saved.',
                     'Active counts are inferred from final output lengths. Completion lags0/1 are sensitivity checks, not observed schedules.',
                     'Padding sizes come from the saved engine log. Actual graph dispatch and inherited BATCH_INVARIANT environment were not saved.'],
        gpu_used=False,extra_model_forwards=0,new_answers=0)
    with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print({k:(v['mismatch_count'],v['modeled_transitions'][0]['first_padded_difference'],v['all_after_modeled_transition']) for k,v in results.items()})


if __name__=='__main__':main()
