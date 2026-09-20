"""Fixed paired analysis; no grading, generation, or threshold selection."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    target=a.output/'analysis.json'
    if target.exists():raise FileExistsError(target)
    names=['U','R','S','RS'];arms={};records={};correct={}
    for name in names:
        result=json.loads((a.output/'screening'/name/'result.json').read_text(encoding='utf-8'))
        grade=json.loads((a.output/'screening'/name/'author_grade.json').read_text(encoding='utf-8'))
        assert result['status']=='complete' and len(result['records'])==len(grade['records'])==64
        records[name]=result['records']
        assert [r['train_index'] for r in records[name]]==[r['train_index'] for r in grade['records']]
        correct[name]=np.array([r['author_correct'] for r in grade['records']])
        arms[name]=dict(correct=int(correct[name].sum()),accuracy=float(correct[name].mean()),
            mean_thinking_tokens=float(np.mean([r['thinking_tokens'] for r in records[name]])),
            mean_total_tokens=float(np.mean([r['tokens'] for r in records[name]])),
            mean_answer_tokens=float(np.mean([r['answer_tokens'] for r in records[name]])),
            capped=sum(r['tokens']==16000 for r in records[name]),
            forced=sum(bool(r['hybrid'] and r['hybrid']['first_trigger']>=0) for r in records[name]),
            generation_seconds=result['generation_seconds'],checkpoint_io_seconds=result['checkpoint_io_seconds'],
            statistics_mask_gpu_seconds=result['control']['statistics_and_mask_gpu_ms']/1000,
            extra_model_forwards=0,probe_tokens=0,author_grade_seconds=grade['seconds'])
    ids=np.array([r['train_index'] for r in records['U']])
    assert all([r['train_index'] for r in records[name]]==ids.tolist() for name in names)
    rng=np.random.default_rng(20260912);sample=rng.integers(0,64,(10000,64))
    def ci(x):return np.quantile(x,[.025,.975]).tolist()
    comparisons={}
    for left,right in itertools.combinations(names,2):
        data={}
        for field in ('tokens','thinking_tokens'):
            x=np.array([r[field] for r in records[left]],dtype=float)
            y=np.array([r[field] for r in records[right]],dtype=float)
            data[field]=dict(delta=float((y-x).mean()),delta_ci95=ci((y-x)[sample].mean(1)),
                percent_change=100*(y.mean()/x.mean()-1),
                percent_change_ci95=ci(100*(y[sample].mean(1)/x[sample].mean(1)-1)))
        change=correct[right].astype(float)-correct[left].astype(float)
        data.update(accuracy_delta_pp=100*float(change.mean()),accuracy_delta_pp_bootstrap_ci95=ci(100*change[sample].mean(1)),
                    wrong_to_right=ids[~correct[left]&correct[right]].tolist(),right_to_wrong=ids[correct[left]&~correct[right]].tolist())
        comparisons[right+'_minus_'+left]=data
    interaction={}
    for field in ('tokens','thinking_tokens'):
        values={n:np.array([r[field] for r in records[n]],dtype=float) for n in names}
        contrast=values['RS']-values['R']-values['S']+values['U']
        interaction[field]=dict(additive=float(contrast.mean()),ci95=ci(contrast[sample].mean(1)))
    gate=all(arms['RS']['mean_thinking_tokens']<arms[n]['mean_thinking_tokens'] and
             arms['RS']['mean_total_tokens']<arms[n]['mean_total_tokens'] and
             arms['RS']['correct']>=arms[n]['correct'] and
             arms['RS']['capped']<=arms[n]['capped'] for n in ('R','S')) and arms['RS']['forced']>0
    target.write_text(json.dumps(dict(arms=arms,comparisons=comparisons,interaction=interaction,
                      screen_gate_pass=bool(gate),confirmation_run=False,
                      bootstrap=dict(repetitions=10000,seed=20260912,unit='paired question'),
                      limitation='Screening only; no independent confirmation or synergy claim; accuracy bootstrap alone cannot certify noninferiority'),
                      ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(arms=arms,interaction=interaction,screen_gate_pass=bool(gate)),indent=2))


if __name__=='__main__':main()
