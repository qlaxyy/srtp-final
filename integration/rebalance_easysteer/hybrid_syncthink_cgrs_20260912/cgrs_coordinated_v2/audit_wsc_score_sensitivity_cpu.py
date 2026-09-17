"""Decompose saved feature/probe differences; never rerun or relabel samples."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from audit_wsc_weights_cpu import PINS,TensorOnlyUnpickler
from wsc_recovery_policy_cpu import LoopState


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-80,80)))


def triggers(scores,lengths):
    state=LoopState()
    return [state.observe(float(s),int(n)) for s,n in zip(scores,lengths)]


def audit(run,weights):
    raw=weights.read_bytes();assert hashlib.sha256(raw).hexdigest()==PINS['1.5B'][0]
    state=TensorOnlyUnpickler(io.BytesIO(raw)).load()['model_state_dict']
    w=state['weight'].numpy().reshape(-1);bias=float(state['bias'][0]);wd=w.astype(np.float64)
    results=[];files=[]
    for idx in (5353,26):
        arrays=[]
        for size in (256,128):
            path=run/f'{idx}_chunk{size}/features.npz'
            files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            with np.load(path,allow_pickle=False) as f:
                arrays.append({k:f[k].copy() for k in ('hidden','positions','chunk_tokens')})
        a,b=arrays
        assert np.array_equal(a['positions'],b['positions'])
        assert np.array_equal(a['chunk_tokens'],b['chunk_tokens'])
        x,y=a['hidden'],b['hidden'];xd,yd=x.astype(np.float64),y.astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        z32x=x@w+bias;z32y=y@w+bias;zx=xd@wd+bias;zy=yd@wd+bias
        sx,sy=sigmoid(zx),sigmoid(zy)
        tx=(torch.from_numpy(xd)@torch.from_numpy(wd)+bias).numpy()
        ty=(torch.from_numpy(yd)@torch.from_numpy(wd)+bias).numpy()
        delta=xd-yd;projected=delta@wd
        assert np.allclose(zx-zy,projected,rtol=1e-10,atol=1e-10)
        j=int(np.argmax(abs(sx-sy)));contrib=delta[j]*wd
        top=np.argsort(-abs(contrib))[:10]
        results.append(dict(train_index=idx,steps=len(sx),positions_exact=True,lengths_exact=True,
            max_fp32_vs_fp64_score_error=float(max(np.max(abs(sigmoid(z32x)-sx)),np.max(abs(sigmoid(z32y)-sy)))),
            max_torch_vs_numpy_fp64_logit_error=float(max(np.max(abs(tx-zx)),np.max(abs(ty-zy)))),
            feature_projection_identity_residual=float(np.max(abs((zx-zy)-projected))),
            max_partition_score_difference_fp64=float(np.max(abs(sx-sy))),
            threshold_side_changes=int(np.sum((sx>.5)!=(sy>.5))),
            trigger_changes=int(np.sum(np.array(triggers(sx,a['chunk_tokens']))!=np.array(triggers(sy,b['chunk_tokens'])))),
            min_observed_distance_to_half=float(min(np.min(abs(sx-.5)),np.min(abs(sy-.5)))),
            max_difference=dict(position=int(a['positions'][j]),score256=float(sx[j]),score128=float(sy[j]),
                logit256=float(zx[j]),logit128=float(zy[j]),delta_logit=float(projected[j]),
                feature_relative_l2=float(np.linalg.norm(delta[j])/np.linalg.norm(xd[j])),
                feature_cosine=float(np.dot(xd[j],yd[j])/(np.linalg.norm(xd[j])*np.linalg.norm(yd[j]))),
                sum_absolute_coordinate_contributions=float(abs(contrib).sum()),
                top_coordinates=[dict(index=int(k),contribution=float(contrib[k])) for k in top]),
            limitations='Same-side values on these saved steps do not bound unseen perturbations or establish semantic correctness.'))
    return dict(results=results,assets=files,probe_sha256=hashlib.sha256(raw).hexdigest(),gpu_used=False,
        llm_forwards=0,old_gate_status='failed_unchanged',
        conclusion='Saved hidden-state differences explain the score drift through the fixed linear probe; FP64 scoring alone does not repair it. This cannot locate the upstream numerical source or establish classifier quality.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.weights)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['results'],indent=2))
