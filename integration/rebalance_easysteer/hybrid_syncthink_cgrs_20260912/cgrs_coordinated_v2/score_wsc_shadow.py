"""Offline CPU scoring of completed engineering captures, never edits outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from audit_wsc_weights_cpu import PINS, TensorOnlyUnpickler
from wsc_feature_contract_cpu import BoundaryLedger
from wsc_recovery_policy_cpu import LoopState


def score_capture(data, boundaries, weight, bias):
    selected=data['selected'];positions=data['positions'];inputs=data['input_ids'];hidden=data['hidden']
    plen=int(data['prompt_tokens'])
    assert len(selected)==len(positions)==len(inputs)==len(hidden)>0
    assert np.array_equal(positions,np.arange(plen-1,plen-1+len(selected)))
    assert np.array_equal(inputs[1:],selected[:-1])
    ledger=BoundaryLedger(plen,len(weight),boundaries);loop=LoopState();rows=[]
    # The first observation is prompt prefill. Last sampled token has not been
    # processed by a subsequent forward and cannot be assigned a hidden state.
    for pos,token,h,accepted in zip(positions,inputs,hidden,selected):
        event=ledger.observe(int(pos),int(token),h)
        if event is not None:
            logit=float(event['hidden']@weight+bias)
            score=float(1/(1+np.exp(-np.clip(logit,-80,80))))
            fired=loop.observe(score,event['chunk_tokens'])
            rows.append(dict(position=event['position'],chunk_tokens=event['chunk_tokens'],
                score=score,would_trigger=fired,long_streak=loop.long,short_streak=loop.short))
        ledger.accept(int(accepted))
    return dict(events=rows,invalid_feature=ledger.disabled,final_sampled_token_unobserved=True,
                generated_tokens=len(selected),would_trigger=loop.fired)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    assert json.loads((a.run/'engineering_gate.json').read_text())['passed']
    raw=a.weights.read_bytes();assert hashlib.sha256(raw).hexdigest()==PINS['1.5B'][0]
    obj=TensorOnlyUnpickler(io.BytesIO(raw)).load();state=obj['model_state_dict']
    w=state['weight'].numpy().reshape(-1);bias=float(state['bias'][0])
    assert w.shape==(1536,) and np.isfinite(w).all() and np.isfinite(bias)
    boundaries=json.loads((a.run/'tokenizer_boundaries.json').read_text())['ids']
    plan=json.loads((a.run/'plan.json').read_text());results={}
    for row in plan['rows']:
        path=a.run/'RC14_wsc_shadow'/f"{row['train_index']}.npz"
        with np.load(path,allow_pickle=False) as data:results[str(row['train_index'])]=score_capture(data,boundaries,w,bias)
    report=dict(results=results,seconds=time.monotonic()-start,device='cpu',
        weight_sha256=hashlib.sha256(raw).hexdigest(),gpu_forward_calls=0,
        scope='Input/score engineering diagnostics only. Positive scores are not validated semantic-loop labels. No efficacy claims.')
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    if any(r['invalid_feature'] for r in results.values()):raise ValueError('Invalid feature; preserve report and stop')
    if not any(r['events'] for r in results.values()):raise ValueError('No scored boundaries; inconclusive, no automatic expansion')


if __name__=='__main__':main()
