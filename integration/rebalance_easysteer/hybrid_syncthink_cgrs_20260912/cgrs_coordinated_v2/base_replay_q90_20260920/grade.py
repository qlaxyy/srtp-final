"""Grade complete arms only; frozen labels are read-only references."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
HOME=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/projects/iterative_recalibration_20260919/integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/iterative_recalibration_20260919')
from common import read,save,sha

def summary(rows):
    return dict(count=len(rows),correct=sum(int(r['correct']) for r in rows),
        accuracy_percent=100*np.mean([r['correct'] for r in rows]),
        mean_tokens=float(np.mean([r['tokens'] for r in rows])),
        mean_thinking_tokens=float(np.mean([r['thinking_tokens'] for r in rows])),
        capped=sum(r['tokens']==16000 for r in rows))

def compare(base,new):
    assert [r['problem_sha256'] for r in base]==[r['problem_sha256'] for r in new]
    delta=np.array([int(b['correct'])-int(a['correct']) for a,b in zip(base,new)])
    arrays={k:(np.array([r[k] for r in base]),np.array([r[k] for r in new])) for k in ('tokens','thinking_tokens')}
    rng=np.random.default_rng(20260917);boots={k:[] for k in ['accuracy','tokens','thinking_tokens']}
    for _ in range(100):
        ix=rng.integers(0,len(base),(200,len(base)));boots['accuracy'].extend(100*delta[ix].mean(1))
        for k,(a,b) in arrays.items():boots[k].extend(100*(b[ix].mean(1)/a[ix].mean(1)-1))
    changes={k:float(100*(b.mean()/a.mean()-1)) for k,(a,b) in arrays.items()}
    return dict(accuracy_delta_pp=float(100*delta.mean()),percent_changes=changes,
        ci95={k:np.quantile(v,[.025,.975]).tolist() for k,v in boots.items()},
        wrong_to_right=[i for i,d in enumerate(delta) if d==1],right_to_wrong=[i for i,d in enumerate(delta) if d==-1],
        point_gate=bool(delta.mean()>=-.02 and all(v<0 for v in changes.values()) and summary(new)['capped']<=summary(base)['capped']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True)
    a=p.parse_args();gate=read(a.run/'complete.json');assert gate['passed'] and not gate['engineering']
    data=read(a.run/'result.json');assert len(data['records'])==500
    refs=read(HOME/'historical_compact.json');rows=read(HOME/'rows.json')
    for n,h in refs['grader_sha256'].items():assert sha(a.runtime_root/'sources/ReBalance/utils'/n)==h,n
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    labels=[];start=time.monotonic()
    with (a.run/'author_labels.jsonl').open('x') as f:
        for i,(r,expected) in enumerate(zip(data['records'],rows)):
            assert r['dataset_index']==i and r['problem_sha256']==expected['problem_sha256']
            assert len(r['token_ids'])==r['tokens']<=16000
            assert r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids']))
            _,gold=parse_ground_truth(expected,'math');answer=extract_answer(r['text'])
            label=dict(dataset_index=i,problem_sha256=r['problem_sha256'],correct=bool(check_is_correct(answer,gold)),
                tokens=r['tokens'],thinking_tokens=r['thinking_tokens'])
            labels.append(label);f.write(json.dumps(label)+'\n');f.flush()
    save(a.run/'analysis.json',dict(summary=summary(labels),comparisons={k:compare(v['records'],labels) for k,v in refs['groups'].items()},
        generation_seconds=data['generation_seconds'],extra_model_forwards=0,grading_seconds=time.monotonic()-start,
        status='Exposed MATH500 exploratory comparison; not independent confirmation',source_sha256=sha(a.run/'result.json')))

if __name__=='__main__':main()
