"""Author grading and paired descriptive statistics; never grade partial arms."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from prepare_label_alignment import save,sha


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def compare(base,candidate):
    assert [r['problem_sha256'] for r in base]==[r['problem_sha256'] for r in candidate]
    n=len(base);rng=np.random.default_rng(20260917)
    delta=np.array([int(b['correct'])-int(a['correct']) for a,b in zip(base,candidate)])
    cis={k:[] for k in ('accuracy','tokens','thinking_tokens')}
    arrays={k:(np.array([r[k] for r in base]),np.array([r[k] for r in candidate])) for k in ('tokens','thinking_tokens')}
    for _ in range(100):
        ix=rng.integers(0,n,(200,n));cis['accuracy'].extend(100*delta[ix].mean(1))
        for k,(a,b) in arrays.items():cis[k].extend(100*(b[ix].mean(1)/a[ix].mean(1)-1))
    intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in cis.items()}
    return dict(accuracy_delta_pp=float(100*delta.mean()),ci95=intervals,
        tokens_percent=float(100*(arrays['tokens'][1].mean()/arrays['tokens'][0].mean()-1)),
        thinking_tokens_percent=float(100*(arrays['thinking_tokens'][1].mean()/arrays['thinking_tokens'][0].mean()-1)),
        wrong_to_right=[r['dataset_index'] for r,d in zip(base,delta) if d==1],
        right_to_wrong=[r['dataset_index'] for r,d in zip(base,delta) if d==-1],
        accuracy_observed_within_2pp=bool(delta.mean()>=-.02),
        nominal_accuracy_ci_within_2pp=bool(intervals['accuracy'][0]>=-2))


def summary(rows):
    return dict(count=len(rows),correct=sum(r['correct'] for r in rows),
        accuracy_percent=100*np.mean([r['correct'] for r in rows]),
        mean_tokens=np.mean([r['tokens'] for r in rows]),
        mean_thinking_tokens=np.mean([r['thinking_tokens'] for r in rows]),
        capped=sum(r['tokens']==16000 for r in rows))


def interaction(groups):
    result={};rng=np.random.default_rng(20260917)
    for key in ('tokens','thinking_tokens','correct'):
        x=np.array([float(a[key])-float(b[key])-float(c[key])+float(d[key]) for a,b,c,d in
            zip(groups['T14_L27'],groups['T14_T14'],groups['L27_L27'],groups['RC14'])])
        if key=='correct':x*=100
        boots=[]
        for _ in range(100):boots.extend(x[rng.integers(0,len(x),(200,len(x)))].mean(1))
        result[key]=dict(difference_in_differences=float(x.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    complete=read(a.run/'complete.json');assert complete['phase']=='full' and complete['passed']
    release=read(a.run/'release.json');plan=read(a.run/'plan.json')
    assert not (a.run/'analysis.json').exists()
    refs_path=Path(__file__).parent/'label_alignment_20260917/historical_compact.json'
    assert sha(refs_path)==release['artifact_sha256']['historical_compact.json']
    refs=read(refs_path);groups={k:v['records'] for k,v in refs['groups'].items()}
    for name,h in refs['grader_sha256'].items():assert sha(a.runtime_root/'sources/ReBalance/utils'/name)==h
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    started=time.monotonic();results={}
    for name in complete['arms']:
        folder=a.run/name;data=read(folder/'result.json');records=data['records']
        assert data['status']=='complete' and len(records)==500
        path=folder/'author_partial.jsonl'
        previous=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if a.resume and path.exists() else []
        labels=[]
        with path.open('a' if a.resume else 'x',encoding='utf8') as f:
            for i,(row,r) in enumerate(zip(plan['rows'],records)):
                assert row['dataset_index']==r['dataset_index']==i and row['problem_sha256']==r['problem_sha256']
                ids=r['token_ids'];assert len(ids)==r['tokens']<=16000
                assert r['thinking_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
                identity=dict(dataset_index=i,problem_sha256=row['problem_sha256'],text_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
                if i<len(previous):
                    label=previous[i];assert all(label[k]==v for k,v in identity.items())
                else:
                    _,gold=parse_ground_truth(row,'math');answer=extract_answer(r['text'])
                    label=dict(identity,correct=bool(check_is_correct(answer,gold)))
                    f.write(json.dumps(label)+'\n');f.flush()
                labels.append(dict(label,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
        assert len(previous)<=500
        groups[name]=labels
        results[name]=dict(summary=summary(labels),comparisons={k:compare(groups[k],labels) for k in ('U','R','RC14')},
            generation_seconds=data['generation_seconds'],setup_seconds=data['setup_seconds'],
            extra_model_forward_count=data['extra_model_forward_count'],control_gpu_seconds=data['control_gpu_seconds'],
            source_result_sha256=sha(folder/'result.json'),labels_sha256=sha(path))
    save(a.run/'analysis.json',dict(status='complete exploratory full benchmark, no independent confirmation',
        candidates=results,references={k:summary(groups[k]) for k in ('U','R','RC14')},
        factorial_interaction=interaction(groups),bootstrap_replicates=20000,
        generation_seed=42,analysis_rng_seed=20260917,grading_and_analysis_seconds=time.monotonic()-started,
        limitations=['Same seed is not numerical equivalence across historical runtimes.',
            'Five candidates and a repeatedly exposed test set: intervals are descriptive and unadjusted; do not select a winner and call it confirmed.',
            'Calibration and inference lexical scopes differ; same inventory alone does not guarantee benefit.']))


if __name__=='__main__':main()
