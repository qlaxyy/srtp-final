"""Question/seed-paired summaries, with all failures and capped outputs retained."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from engineering import ROOT,read,save,sha
from repeatability import ARMS,SEEDS,validate,point_decision

def compare(a,b,bootstrap=10000):
    # Matrices [seed, question, correct/total/thinking/capped]. Paired on both axes.
    assert a.shape==b.shape and a.ndim==3 and a.shape[2]==4
    def metrics(x,y):
        return [100*(y[...,0].mean()-x[...,0].mean()),100*(y[...,1].mean()/x[...,1].mean()-1),100*(y[...,2].mean()/x[...,2].mean()-1)]
    point=metrics(a,b);rng=np.random.default_rng(20260916);conditional=[];crossed=[]
    for _ in range(bootstrap):
        q=rng.integers(0,a.shape[1],a.shape[1]);s=rng.integers(0,a.shape[0],a.shape[0])
        conditional.append(metrics(a[:,q],b[:,q]));crossed.append(metrics(a[s][:,q],b[s][:,q]))
    def ci(values):return dict(zip(('accuracy_delta_pp','total_change_percent','thinking_change_percent'),np.quantile(values,[.025,.975],axis=0).T.tolist()))
    conditional=ci(conditional);crossed=ci(crossed)
    return dict(accuracy_delta_pp=point[0],total_change_percent=point[1],thinking_change_percent=point[2],cap_delta=int(b[...,3].sum()-a[...,3].sum()),
        per_seed=[dict(seed=seed,accuracy_delta_pp=m[0],total_change_percent=m[1],thinking_change_percent=m[2],cap_delta=int(b[i,:,3].sum()-a[i,:,3].sum()),
                      wrong_to_right=np.flatnonzero((a[i,:,0]==0)&(b[i,:,0]==1)).tolist(),right_to_wrong=np.flatnonzero((a[i,:,0]==1)&(b[i,:,0]==0)).tolist()) for i,seed in enumerate(SEEDS) for m in [metrics(a[i],b[i])]],
        question_cluster_ci95=conditional,crossed_question_seed_ci95=crossed,
        interval_support=all(c['accuracy_delta_pp'][0]>=-2 and c['total_change_percent'][1]<0 and c['thinking_change_percent'][1]<0 for c in (conditional,crossed)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    plan=read(out/'resolved_plan.json');validate(plan);status=read(out/'batch_status.json');assert status['passed'] and status['phase']=='screen'
    assert status['plan_sha256']==sha(out/'resolved_plan.json') and not (out/'analysis.json').exists()
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n,True)==h,n
    sys.path.insert(0,str(ROOT/'sources/ReBalance'))
    from utils.parser import extract_answer,parse_ground_truth
    from utils.grader import check_is_correct
    started=time.perf_counter();arrays={arm:np.zeros((3,100,4)) for arm in ARMS};summaries={}
    for case in plan['schedule']:
        folder=out/case['name'];result=read(folder/'result.json');assert result['status']=='complete' and result['sampling_seed']==case['seed']
        records=result['records'];assert len(records)==100
        assert sorted(r['dataset_index'] for r in records)==list(range(100))
        labels=[];si=SEEDS.index(case['seed']);arm=case['arm']
        with (folder/'author_partial.jsonl').open('x',encoding='utf8') as f:
            for rec in records:
                i=rec['dataset_index'];row=plan['rows'][i];assert rec['problem_sha256']==row['problem_sha256'] and rec['problem']==row['problem'] and rec['answer']==row['answer']
                ids=rec['token_ids'];assert len(ids)==rec['tokens']<=16000 and rec['finish_reason'] in ('length','stop')
                assert rec['thinking_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
                _,gold=parse_ground_truth(row,'math');answer=extract_answer(rec['text'],'math');correct=bool(check_is_correct(answer,gold))
                label=dict(dataset_index=i,train_index=row['train_index'],problem_sha256=row['problem_sha256'],correct=correct,extracted_answer=answer,text_sha256=hashlib.sha256(rec['text'].encode()).hexdigest())
                f.write(json.dumps(label,ensure_ascii=False)+'\n');f.flush();labels.append(label)
                arrays[arm][si,i]=[correct,rec['tokens'],rec['thinking_tokens'],rec['finish_reason']=='length']
        a=arrays[arm][si];completion=sorted(r['completed_seconds'] for r in records)
        summaries[case['name']]=dict(n=100,correct=int(a[:,0].sum()),accuracy_percent=float(a[:,0].mean()*100),mean_total_tokens=float(a[:,1].mean()),mean_thinking_tokens=float(a[:,2].mean()),capped=int(a[:,3].sum()),generation_seconds=result['generation_seconds'],last10_completion_span_seconds=completion[-1]-completion[89],checkpoint_io_seconds=result['checkpoint_io_seconds'],setup_seconds=result['setup_seconds'],extra_probe_forwards=0,control_gpu_seconds=None,control_note='Inline adapter cost included in generation; no separate CUDA timing to avoid hot-path synchronization.',scheduler_preemptions=len(result['scheduler_preemptions']),replay_counts=result['replay_counts'],result_sha256=sha(folder/'result.json'),labels=labels)
        values=[float(x[1]) for x in csv.reader((folder/'gpu.csv').read_text().splitlines()) if len(x)==4]
        summaries[case['name']]['gpu_mean_percent']=float(np.mean(values)) if values else None
    comparisons={b+'_vs_'+a:compare(arrays[a],arrays[b]) for a,b in [('R','RC14'),('R','RChistory'),('RC14','RChistory')]}
    for key,c in comparisons.items():
        for s in c['per_seed']:
            for kind in ('wrong_to_right','right_to_wrong'):
                s[kind]=[dict(dataset_index=i,train_index=plan['rows'][i]['train_index'],problem_sha256=plan['rows'][i]['problem_sha256']) for i in s[kind]]
    aggregate={arm:dict(unique_questions=100,seeds=3,observations=300,correct=int(a[...,0].sum()),accuracy_percent=float(100*a[...,0].mean()),mean_total_tokens=float(a[...,1].mean()),mean_thinking_tokens=float(a[...,2].mean()),capped=int(a[...,3].sum()),generation_seconds=sum(summaries[f's{seed}_{arm}']['generation_seconds'] for seed in SEEDS)) for arm,a in arrays.items()}
    save(out/'analysis.json',dict(status='complete_exposed_training_repeatability',groups=summaries,aggregate=aggregate,comparisons=comparisons,decision=point_decision(comparisons),grade_seconds=time.perf_counter()-started,bootstrap_draws=10000,plan_sha256=sha(out/'resolved_plan.json'),limitations=plan['limitations']))

if __name__=='__main__':main()
