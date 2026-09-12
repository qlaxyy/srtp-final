"""Prespecified per-dataset confirmation; paired resampling includes failures."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np


def cp_upper(k,n,alpha=.025):
    if k==n:return 1.
    lo,hi=0.,1.
    for _ in range(80):
        p=(lo+hi)/2
        cdf=sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
        if cdf>alpha:lo=p
        else:hi=p
    return (lo+hi)/2


def analyze(root,role):
    names=['U','R','S','RS'];arrays={};arms={};raws={};hashes={}
    for name in names:
        folder=root/role/name
        raw=(folder/'result.json').read_bytes();graw=(folder/'author_grade.json').read_bytes()
        result=json.loads(raw);grade=json.loads(graw)
        assert grade['input_sha256']==hashlib.sha256(raw).hexdigest()
        rows=result['records'];labels=grade['records'];assert len(rows)==len(labels)==200
        assert [r['train_index'] for r in rows]==[r['train_index'] for r in labels]
        assert result['status']=='complete' and result['cap']==16000
        arrays[name]={k:np.array([r[k] for r in rows]) for k in ('tokens','thinking_tokens','answer_tokens')}
        arrays[name]['correct']=np.array([r['author_correct'] for r in labels],dtype=int)
        raws[name]=rows
        arms[name]=dict(correct=int(arrays[name]['correct'].sum()),accuracy=float(arrays[name]['correct'].mean()),
            **{'mean_'+k:float(arrays[name][k].mean()) for k in ('tokens','thinking_tokens','answer_tokens')},
            capped=int((arrays[name]['tokens']==16000).sum()),
            capped_ids=[r['train_index'] for r in rows if r['tokens']==16000],
            forced=sum(bool(r['hybrid'] and r['hybrid']['first_trigger']>=0) for r in rows),
            discarded_async_tail_tokens=sum((r['hybrid'] or {}).get('discarded_async_tail_tokens',0) for r in rows),
            generation_seconds=result['generation_seconds'],checkpoint_io_seconds=result['checkpoint_io_seconds'],
            statistics_mask_gpu_seconds=result['control']['statistics_and_mask_gpu_ms']/1000,
            author_grade_seconds=grade['seconds'],extra_model_forwards=0,probe_tokens=0)
        hashes[name]=dict(result=hashlib.sha256(raw).hexdigest(),grade=hashlib.sha256(graw).hexdigest())
    ids=np.array([r['train_index'] for r in raws['U']])
    assert all([r['train_index'] for r in raws[n]]==ids.tolist() for n in names)
    sample=np.random.default_rng(20260912).integers(0,200,(10000,200))
    def ci(x,level=.95):return np.quantile(x,[(1-level)/2,(1+level)/2]).tolist()
    comparisons={}
    for left,right in itertools.combinations(names,2):
        data={}
        for k in ('tokens','thinking_tokens'):
            x,y=arrays[left][k],arrays[right][k]
            deltas=(y-x)[sample].mean(1)
            data[k]=dict(delta=float((y-x).mean()),delta_ci95=ci(deltas),delta_ci97_5=ci(deltas,.975),
                percent_change=float(100*(y.mean()/x.mean()-1)),percent_change_ci95=ci(100*(y[sample].mean(1)/x[sample].mean(1)-1)))
        x,y=arrays[left]['correct'],arrays[right]['correct']
        harmed=(x==1)&(y==0)
        data.update(accuracy_delta_pp=float(100*(y-x).mean()),accuracy_delta_pp_ci95=ci(100*(y-x)[sample].mean(1)),wrong_to_right=ids[(x==0)&(y==1)].tolist(),right_to_wrong=ids[harmed].tolist(),gross_harm_probability_upper97_5=cp_upper(int(harmed.sum()),200))
        comparisons[right+'_minus_'+left]=data
    interaction={}
    for k in ('tokens','thinking_tokens','correct'):
        d=arrays['RS'][k]-arrays['R'][k]-arrays['S'][k]+arrays['U'][k]
        interaction[k]=dict(additive=float(d.mean()),ci95=ci(d[sample].mean(1)))
    point=all(arms['RS']['mean_tokens']<arms[n]['mean_tokens'] and arms['RS']['mean_thinking_tokens']<arms[n]['mean_thinking_tokens'] and arms['RS']['correct']>=arms[n]['correct'] and arms['RS']['capped']<=arms[n]['capped'] for n in ('R','S')) and arms['RS']['forced']>0
    precise=point and all(comparisons['RS_minus_'+n]['tokens']['delta_ci97_5'][1]<0 and comparisons['RS_minus_'+n]['gross_harm_probability_upper97_5']<=.02 for n in ('R','S'))
    return dict(dataset=role,n=200,arms=arms,comparisons=comparisons,interaction=interaction,point_gate_pass=bool(point),confirmation_standard_pass=bool(precise),superadditive_token_evidence=bool(interaction['tokens']['ci95'][1]<0),source_sha256=hashes,bootstrap=dict(repetitions=10000,seed=20260912,unit='paired question'),limitations='One generation seed; MATH and GSM8K reported separately, no pooled rescue; gross-harm bound is conservative, stricter than net accuracy noninferiority; runtime differs from screening; intervals do not include model pretraining leakage or all-method selection uncertainty.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert abs(cp_upper(0,200)-(1-.025**(1/200)))<1e-12
    for role in ('math','gsm8k'):
        value=analyze(a.output,role)
        with (a.output/(role+'_analysis.json')).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
        print(json.dumps(dict(dataset=role,arms=value['arms'],point_gate_pass=value['point_gate_pass'],confirmation_standard_pass=value['confirmation_standard_pass']),ensure_ascii=False))
