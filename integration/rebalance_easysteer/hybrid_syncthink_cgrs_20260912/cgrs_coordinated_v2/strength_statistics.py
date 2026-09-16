"""Shared question resampling across four arms, never extra generation seeds."""
import numpy as np
from strength_screen import ARMS


def comparisons(raw,draws=20000,seed=20260917):
    assert set(raw)==set(ARMS)
    n=len(raw['R']);assert n>0 and all(len(raw[a])==n for a in ARMS)
    identities=[r['problem_sha256'] for r in raw['R']]
    assert len(set(identities))==n
    assert all([r['problem_sha256'] for r in raw[a]]==identities for a in ARMS)
    x=np.array([[[r['correct'],r['tokens'],r['thinking_tokens']] for r in raw[a]] for a in ARMS],dtype=float)
    assert np.isfinite(x).all() and np.all(x[:,:,1]>0) and np.all(x[:,:,2]>=0)
    means=[];rng=np.random.default_rng(seed)
    for start in range(0,draws,500):
        idx=rng.integers(0,n,(min(500,draws-start),n))
        means.append(x[:,idx,:].mean(axis=2))
    resampled=np.concatenate(means,axis=1);point=x.mean(axis=1);result={}
    def difference(a,b):
        value=np.empty_like(a);value[...,0]=100*(b[...,0]-a[...,0])
        with np.errstate(divide='ignore',invalid='ignore'):
            value[...,1:]=100*(b[...,1:]/a[...,1:]-1)
        return value
    for i in range(4):
        for j in range(i+1,4):
            a,b=ARMS[i],ARMS[j];boot=difference(resampled[i],resampled[j]);obs=difference(point[i],point[j])
            intervals={};undefined={}
            keys=('accuracy_delta_pp','total_change_percent','thinking_change_percent')
            for k,key in enumerate(keys):
                undefined[key]=int((~np.isfinite(boot[:,k])).sum())
                intervals[key]=np.quantile(boot[:,k],[.025,.975]).tolist() if not undefined[key] else None
            identity=lambda r:{k:r[k] for k in ('dataset_index','train_index','problem_sha256')}
            result[b+'_vs_'+a]=dict(point={k:float(v) if np.isfinite(v) else None for k,v in zip(keys,obs)},
                paired_ci95=intervals,undefined_bootstrap_draws=undefined,
                wrong_to_correct=[identity(r) for r,s in zip(raw[a],raw[b]) if not r['correct'] and s['correct']],
                correct_to_wrong=[identity(r) for r,s in zip(raw[a],raw[b]) if r['correct'] and not s['correct']],
                capped_delta=sum(r['finish_reason']=='length' for r in raw[b])-sum(r['finish_reason']=='length' for r in raw[a]))
    return result
