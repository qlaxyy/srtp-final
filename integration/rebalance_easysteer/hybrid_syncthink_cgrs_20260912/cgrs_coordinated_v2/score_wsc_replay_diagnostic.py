"""CPU-only report of fixed probe diagnosis; no threshold selection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from audit_wsc_weights_cpu import PINS, TensorOnlyUnpickler
from wsc_recovery_policy_cpu import LoopState


def score(hidden,weight,bias,lengths):
    values=1/(1+np.exp(-np.clip(hidden@weight+bias,-80,80)))
    state=LoopState();events=[]
    for v,n in zip(values,lengths):events.append(state.observe(float(v),int(n)))
    return values,np.array(events,dtype=bool)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads((a.run/'plan.json').read_text());assert (a.run/'complete.json').exists()
    raw=a.weights.read_bytes();assert hashlib.sha256(raw).hexdigest()==PINS['1.5B'][0]
    st=TensorOnlyUnpickler(io.BytesIO(raw)).load()['model_state_dict']
    w=st['weight'].numpy().reshape(-1);bias=float(st['bias'][0]);reports=[];partition=[]
    for case in plan['cases']:
        idx=case['train_index']
        with np.load(a.run/f'{idx}_chunk256/features.npz',allow_pickle=False) as f:
            h=f['hidden'];pos=f['positions'];lengths=f['chunk_tokens']
            assert np.isfinite(h).all() and h.shape==(len(pos),1536)
            values,trigger=score(h,w,bias,lengths)
            events=[dict(position=int(i),chunk_tokens=int(n),score=float(v),would_trigger=bool(t))
                    for i,n,v,t in zip(pos,lengths,values,trigger)]
            reports.append(dict(train_index=idx,annotation=case['annotation'],events=events,
                post_review_start_boundaries=int((pos>=case['review_start']).sum()),
                trigger_positions=[int(i) for i,t in zip(pos,trigger) if t]))
            if idx in plan['partition_checks']:
                with np.load(a.run/f'{idx}_chunk128/features.npz',allow_pickle=False) as g:
                    assert np.array_equal(pos,g['positions']) and np.array_equal(lengths,g['chunk_tokens'])
                    other=g['hidden'];vv,tt=score(other,w,bias,lengths)
                    delta=float(np.max(np.abs(values-vv)))
                    cos=np.sum(h*other,axis=1)/(np.linalg.norm(h,axis=1)*np.linalg.norm(other,axis=1))
                    passed=bool(delta<=.02 and cos.min()>=.999 and np.array_equal(values>.5,vv>.5) and np.array_equal(trigger,tt))
                    partition.append(dict(train_index=idx,max_score_difference=delta,min_cosine=float(cos.min()),passed=passed))
    passed=all(r['passed'] for r in partition) and len(partition)==2
    report=dict(partition_gate=passed,partition=partition,cases=reports,
        interpretation='Qualitative exposed unsteered diagnostics. No accuracy/compression or RC14-domain claim. Review individual labeled spans; never pool ambiguous or boundary-blind cases into classifier sensitivity.',
        next_step='If partition gate fails, stop interpretation and preserve output. If clear observable loops are missed or productive spans trigger, do not tune layer/threshold; reassess probe. Passing does not authorize recovery or full tests.')
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    if not passed:raise ValueError('Chunk partition gate failed; no semantic conclusions')


if __name__=='__main__':main()
