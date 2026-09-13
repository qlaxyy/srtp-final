"""Full RCnegative author grading, resumable without regenerating answers."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from engineering import ROOT,read,save,sha


def compare(ref,records,labels):
    n=len(ref);rng=np.random.default_rng(20260913)
    delta=np.array([int(c['correct'])-int(r['correct']) for r,c in zip(ref,labels)])
    assert [r['problem_sha256'] for r in ref]==[r['problem_sha256'] for r in labels]
    boot_accuracy=[];boot_lengths={k:[] for k in ('tokens','thinking_tokens')}
    # Bound bootstrap memory, using a single deterministic paired-index stream.
    for _ in range(100):
        ix=rng.integers(0,n,size=(100,n));boot_accuracy.extend(100*delta[ix].mean(1))
        for key in boot_lengths:
            a=np.array([r[key] for r in ref],float);b=np.array([r[key] for r in records],float)
            boot_lengths[key].extend(100*(b[ix].mean(1)/a[ix].mean(1)-1))
    ci=lambda x:np.quantile(x,[.025,.975],method='linear').tolist()
    interval=ci(boot_accuracy)
    return dict(accuracy_delta_pp=float(100*delta.mean()),accuracy_ci95=interval,
        point_loss_within_2pp=bool(100*delta.mean()>=-2),nominal_ci_noninferiority_2pp=bool(interval[0]>=-2),
        wrong_to_right=[r['dataset_index'] for r,d in zip(ref,delta) if d==1],
        right_to_wrong=[r['dataset_index'] for r,d in zip(ref,delta) if d==-1],
        lengths={k:dict(percent_change=100*(sum(r[k] for r in records)/sum(r[k] for r in ref)-1),ci95=ci(boot_lengths[k])) for k in boot_lengths})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args();folder=args.output
    plan=read(folder/'resolved_plan.json');history=read(folder/'historical_reference.json')
    assert read(folder/'batch_status.json')['status']=='complete' and not (folder/'analysis.json').exists()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name,True)==digest,name
    sys.path.insert(0,str(ROOT/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    started=time.perf_counter();summaries={};reuse={}
    for role,sub in plan['datasets'].items():
        location=folder/role/'RCnegative';data=read(location/'result.json');records=data['records'];labels=[]
        assert data['status']=='complete' and len(records)==len(sub['rows'])
        partial=location/'author_partial.jsonl'
        previous=[json.loads(s) for s in partial.read_text(encoding='utf8').splitlines()] if args.resume and partial.exists() else []
        assert len(previous)<=len(records)
        reuse[role]=dict(count=len(previous),sha256=sha(partial) if partial.exists() else None)
        with partial.open('a' if args.resume else 'x',encoding='utf8') as f:
            for i,(row,rec) in enumerate(zip(sub['rows'],records)):
                assert row['problem_sha256']==rec['problem_sha256'] and rec['dataset_index']==i
                ids=rec['token_ids'];assert len(ids)==rec['tokens']<=16000
                assert rec['thinking_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
                identity=dict(dataset_index=i,problem_sha256=row['problem_sha256'],text_sha256=hashlib.sha256(rec['text'].encode()).hexdigest())
                if i<len(previous):
                    label=previous[i];assert all(label[k]==v for k,v in identity.items()) and type(label['correct']) is bool
                else:
                    _,gold=parse_ground_truth(row,sub['dataset']);answer=extract_answer(rec['text'],sub['dataset'])
                    label=dict(identity,correct=bool(check_is_correct(answer,gold)))
                    f.write(json.dumps(label)+'\n');f.flush()
                labels.append(label)
                if (i+1)%100==0 or i+1==len(records):print('GRADED',role,i+1,flush=True)
        comparisons={a:compare(d['records'],records,labels) for a,d in history[role]['groups'].items()}
        values=[]
        for row in csv.reader((location/'gpu.csv').read_text().splitlines()):
            if len(row)==4:
                try:values.append(float(row[1]))
                except ValueError:pass
        caps=sum(r['finish_reason']=='length' for r in records)
        rsummary=history[role]['groups']['R']['summary'];c=comparisons['R']
        summaries[role]=dict(n=len(records),correct=sum(x['correct'] for x in labels),
            accuracy_percent=100*sum(x['correct'] for x in labels)/len(records),
            mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/len(records),
            mean_total_tokens=sum(r['tokens'] for r in records)/len(records),capped=caps,
            generation_seconds=data['generation_seconds'],setup_seconds=data['setup_seconds'],
            historical_R_generation_seconds=rsummary['generation_seconds'],
            historical_R_timing_note=rsummary.get('timing_note'),
            time_change_vs_historical_R_percent=(None if rsummary.get('resumed_completed_answers',0) else
                100*(data['generation_seconds']/rsummary['generation_seconds']-1)),
            comparisons=comparisons,grades=labels,
            gpu_utilization=dict(samples=len(values),mean_percent=float(np.mean(values)) if values else None,
                median_percent=float(np.median(values)) if values else None,max_percent=max(values) if values else None),
            costs=dict(probes=0,extra_model_forwards=0,control_gpu_seconds=None,
                checkpoint_io_seconds=data['checkpoint_io_seconds'],
                native_kv_replays=data.get('replay_counts',{}),
                replayed_input_tokens=sum(e.get('replay_prefill_tokens',0) for e in data.get('replay_events',[])),
                eligible=sum(e['eligible'] for e in data['events'].values()),
                interventions=sum(e['changed'] for e in data['events'].values()),
                intervened_questions=sum(e['changed']>0 for e in data['events'].values())),
            passes_fixed_point_criteria=bool(c['point_loss_within_2pp'] and caps<=rsummary['capped'] and all(m['percent_change']<0 for m in c['lengths'].values())),
            result_sha256=sha(location/'result.json'))
    save(folder/'analysis.json',dict(status='complete_exposed_full_test_descriptive',datasets=summaries,
        reused_labels=reuse,grade_invocation_seconds=time.perf_counter()-started,
        grade_time_note='This invocation only; interrupted invocation costs additional if resumed',
        generation=read(folder/'batch_status.json'),plan_sha256=sha(folder/'resolved_plan.json'),
        bootstrap=dict(repetitions=10000,seed=20260913,percentile='linear'),
        limitations=['No fresh U/R/C controls; no factorial synergy or superiority over both components claim',
          'Previously exposed public tests, not untouched independent confirmation; no parameter search on these outcomes',
          'Historical R timing, not simultaneous repeated hardware benchmark',
          'Single-seed paired question bootstrap does not include batch-induced sampling variability',
          'Control GPU kernels not separately profiled; probes and extra model forwards are zero',
          'FINAL_ONLY retains completed partials; all errors and caps included in completed evaluation']))


if __name__=='__main__':main()
