"""Synthetic mechanism and rejection checks. These are not model efficacy results."""
import argparse
from pathlib import Path
import time
import numpy as np
from prepare_bcc import read,save,sha,require
from replay_bcc_prefixes import preflight,compare
from fit_bcc import diagnostic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Output already exists');started=time.perf_counter()
    plan,rows=preflight(a.prepared)
    rng=np.random.default_rng(818);h=np.zeros((76,2,2,1536),dtype=np.float32)
    deltas=rng.normal(0,.05,(76,1536));deltas[:,0]=4
    for i,r in enumerate(rows):
        long=next(j for j,prefix in enumerate(r['prefixes']) if prefix['arm']!=r['short_arm'])
        h[i,long]=deltas[i]
    report,d,null=diagnostic(rows,h,plan['parent_vector_norm'])
    require(report['fixed_numeric_gate'] and report['development_positive']==36,'Known signal rejected')
    require(abs(np.linalg.norm(d)-plan['parent_vector_norm'])<1e-4,'Direction norm mismatch')
    require(abs(np.linalg.norm(null)-plan['parent_vector_norm'])<1e-4,'Null norm mismatch')
    neg=h.copy()
    for i,r in enumerate(rows):
        if r['split']=='development':neg[i]*=-1
    negative,_,_=diagnostic(rows,neg,plan['parent_vector_norm'])
    require(not negative['fixed_numeric_gate'] and negative['development_positive']==0,'Reversed holdout accepted')
    failures=[]
    for name,value in [('zero',np.zeros_like(h)),('nonfinite',h.copy())]:
        if name=='nonfinite':value[0,0,0,0]=np.nan
        try:diagnostic(rows,value,plan['parent_vector_norm'])
        except ValueError:failures.append(name)
        else:raise ValueError('Bad states accepted: '+name)
    x=np.full((2,1536),100.);y=x.copy();y[0,0]+=.125
    compare(x,y)
    y[0,0]+=.001
    try:compare(x,y)
    except ValueError:failures.append('absolute_tolerance')
    else:raise ValueError('Relaxed numerical gate')
    # The optimized Gram projection must agree with explicit vectors on independent signs.
    fit=np.array([i for i,r in enumerate(rows) if r['split']=='fit']);dev=np.array([i for i,r in enumerate(rows) if r['split']=='development'])
    cross=deltas[fit]@deltas[dev].T
    for _ in range(100):
        s=rng.choice([-1.,1.],40)
        require(np.allclose((s@deltas[fit])@deltas[dev].T,s@cross,rtol=1e-11,atol=1e-11),'Gram mismatch')
    save(a.output,dict(status='passed_synthetic_CPU_only',plan_sha256=sha(a.prepared/'plan.json'),
        synthetic_positive=report,synthetic_negative=negative,expected_rejections=failures,
        gram_checks=100,seconds=time.perf_counter()-started,new_answers=0,model_forwards=0,
        GPU_calls=0,limitation='Synthetic implementation checks; no real hidden states or compression benefit measured'))
    print('Passed: real prefix preflight, synthetic signal/reversal, norm, bad-state rejection, tolerance, 100 Gram checks')


if __name__=='__main__':main()

