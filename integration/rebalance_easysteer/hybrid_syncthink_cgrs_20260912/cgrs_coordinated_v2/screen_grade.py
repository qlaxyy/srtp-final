"""Resumable author grading; fixed paired descriptive screening statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from engineering import ROOT, read, save, sha
from screen import ARMS


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def analyze(results, grades):
    n = len(grades['R'])
    assert n == 64
    keys = [(r['train_index'], r['problem_sha256']) for r in results['R']['records']]
    for a in ARMS:
        assert [(r['train_index'], r['problem_sha256']) for r in results[a]['records']] == keys
        assert [(r['train_index'], r['problem_sha256']) for r in grades[a]] == keys
    ids = np.random.default_rng(20260913).integers(0, n, size=(10000,n))
    def ci(values):
        # Explicit percentile convention fixed before outcomes.
        return np.quantile(values, [.025,.975], method='linear').tolist()
    summary = {}
    for a in ARMS:
        data=results[a]; rows=data['records']
        summary[a] = dict(n=n, correct=sum(r['correct'] for r in grades[a]),
            accuracy_percent=100*sum(r['correct'] for r in grades[a])/n,
            mean_thinking_tokens=np.mean([r['thinking_tokens'] for r in rows]).item(),
            mean_total_tokens=np.mean([r['tokens'] for r in rows]).item(),
            capped=sum(r['finish_reason']=='length' for r in rows),
            generation_seconds=data['generation_seconds'], setup_seconds=data['setup_seconds'],
            eligible=sum(e['eligible'] for e in data['events'].values()),
            interventions=sum(e['changed'] for e in data['events'].values()),
            extra_probe_count=0, extra_model_forward_count=0,
            control_gpu_seconds=data['control_gpu_seconds'],
            checkpoint_io_seconds=data['checkpoint_io_seconds'])
    comparisons = {}
    for cand, ref in [('RCnegative','R'),('RCnegative','Clex'),('RCnegative','RCalways'),
                      ('RCalways','R'),('RCalways','Clex'),('R','U'),('Clex','U')]:
        c,r=grades[cand],grades[ref]
        delta=np.array([int(x['correct'])-int(y['correct']) for x,y in zip(c,r)])
        metrics={}
        for key in ('tokens','thinking_tokens'):
            x=np.array([v[key] for v in results[cand]['records']],dtype=float)
            y=np.array([v[key] for v in results[ref]['records']],dtype=float)
            metrics[key]=dict(percent_change=100*(x.mean()/y.mean()-1),
                ci95=ci(100*(x[ids].mean(axis=1)/y[ids].mean(axis=1)-1)))
        interval=ci(100*delta[ids].mean(axis=1))
        comparisons[cand+'_minus_'+ref]=dict(accuracy_delta_pp=100*delta.mean(),
            accuracy_ci95=interval, point_loss_within_2pp=bool(100*delta.mean()>=-2),
            nominal_ci_establishes_2pp_noninferiority=bool(interval[0]>=-2),
            wrong_to_right=[keys[i][0] for i,d in enumerate(delta) if d==1],
            right_to_wrong=[keys[i][0] for i,d in enumerate(delta) if d==-1], lengths=metrics)
    interactions={}
    for key in ('tokens','thinking_tokens'):
        value=lambda a: np.array([r[key] for r in results[a]['records']],dtype=float)
        delta=value('RCalways')-value('R')-value('Clex')+value('U')
        interactions[key]=dict(mean_token_interaction=delta.mean().item(),ci95=ci(delta[ids].mean(axis=1)))
    primary=comparisons['RCnegative_minus_R']
    keep=(primary['point_loss_within_2pp'] and
          all(x['percent_change']<0 for x in primary['lengths'].values()) and
          summary['RCnegative']['capped']<=summary['R']['capped'])
    return dict(status='screening_only_not_confirmation', arms=summary,
        comparisons=comparisons, interaction_RCalways_minus_R_minus_Clex_plus_U=interactions,
        primary_candidate='RCnegative', primary_passes_point_screen=bool(keep),
        decision='Eligible for separately authorized independent confirmation' if keep else
                 'Stop fixed primary candidate; do not retune on these rows or automatically expand',
        bootstrap=dict(seed=20260913,repetitions=10000,percentile_method='linear'),
        limitations=['Single seed; question bootstrap does not include batch/runtime variability',
          'Nominal intervals, no multiplicity correction; primary comparison fixed in advance',
          '64 questions: one correct answer is 1.5625pp; point 2pp margin is not proof of noninferiority',
          'Clex is borrowed lexical ablation, not official CGRS',
          'Primary RCnegative gate differs from factorial RCalways; do not apply its interaction estimate to gated candidate',
          'GPU control kernels are not separately timed; end-to-end cost and operation counts are reported',
          'No throughput repetition or full-load claim; diagnostic arm cannot replace primary after observing outcomes'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args();folder=args.output
    plan=read(folder/'resolved_plan.json')
    assert read(folder/'batch_status.json')['status']=='complete' and plan['phase']=='screen'
    assert not (folder/'analysis.json').exists()
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name,True)==digest,name
    sys.path.insert(0,str(ROOT/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    started=time.perf_counter();results={};grades={};reused={}
    for arm in ARMS:
        data=read(folder/arm/'result.json');results[arm]=data
        assert data['status']=='complete' and len(data['records'])==64
        partial=folder/arm/'author_partial.jsonl'
        previous=[json.loads(x) for x in partial.read_text(encoding='utf8').splitlines()] if args.resume and partial.exists() else []
        assert len(previous)<=64
        reused[arm]=dict(count=len(previous),sha256=sha(partial) if partial.exists() else None)
        labels=[]
        with partial.open('a' if args.resume else 'x',encoding='utf8') as stream:
            for i,(row,rec) in enumerate(zip(plan['rows'],data['records'])):
                assert row['problem_sha256']==rec['problem_sha256']
                assert row['train_index']==rec['train_index']
                assert rec['tokens']==len(rec['token_ids'])<=16000
                assert rec['thinking_tokens']==(rec['token_ids'].index(151649) if 151649 in rec['token_ids'] else rec['tokens'])
                identity=dict(train_index=row['train_index'],problem_sha256=row['problem_sha256'],
                              text_sha256=text_hash(rec['text']))
                if i<len(previous):
                    label=previous[i]
                    assert all(label[k]==v for k,v in identity.items()) and type(label['correct']) is bool
                else:
                    _,gold=parse_ground_truth(row,'math');answer=extract_answer(rec['text'],'math')
                    label=dict(identity,correct=bool(check_is_correct(answer,gold)))
                    stream.write(json.dumps(label)+'\n');stream.flush()
                labels.append(label)
        grades[arm]=labels
    report=analyze(results,grades)
    for arm in ARMS:
        values=[]
        for row in csv.reader((folder/arm/'gpu.csv').read_text().splitlines()):
            if len(row)==4:
                try:values.append(float(row[1]))
                except ValueError:pass
        report['arms'][arm]['gpu_utilization']=dict(samples=len(values),
            mean_percent=float(np.mean(values)) if values else None,
            note='Per-arm monitor envelope includes process start/stop around generation')
    report.update(grades=grades,reused_labels=reused,grade_invocation_seconds=time.perf_counter()-started,
        grade_timing_note='Current invocation only; retain interruption costs separately',
        inputs_sha256={a:sha(folder/a/'result.json') for a in ARMS},
        plan_sha256=sha(folder/'resolved_plan.json'),implementation_sha256=sha(Path(__file__)))
    save(folder/'analysis.json',report)


if __name__=='__main__':main()
