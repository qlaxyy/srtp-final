"""Fixed BCC fit and development diagnostic; no tuning, model or generation."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from prepare_bcc import read, save, sha, require


def diagnostic(records, states, parent_norm, permutations=10000):
    require(states.shape==(76,2,2,1536), 'Expected 76 pairs x 2 arms x 2 positions x 1536')
    require(np.isfinite(states).all(), 'Nonfinite states')
    delta=[]
    for r,h in zip(records,states):
        short=next(i for i,p in enumerate(r['prefixes']) if p['arm']==r['short_arm'])
        delta.append(h[1-short].astype(np.float64).mean(0)-h[short].astype(np.float64).mean(0))
    delta=np.asarray(delta); fit=np.array([i for i,r in enumerate(records) if r['split']=='fit'])
    dev=np.array([i for i,r in enumerate(records) if r['split']=='development'])
    require(len(fit)==40 and len(dev)==36, 'Wrong fixed split')
    raw=delta[fit].mean(0); threshold=1e-6*np.median(np.linalg.norm(delta[fit],axis=1))
    signs=np.array([records[i]['placebo_sign'] for i in fit])
    null=(delta[fit]*signs[:,None]).mean(0)
    require(np.linalg.norm(raw)>max(threshold,0) and np.linalg.norm(null)>max(threshold,0), 'Degenerate direction')
    direction=raw/np.linalg.norm(raw)*parent_norm; placebo=null/np.linalg.norm(null)*parent_norm
    counts={batch:int(sum(direction@delta[i]>0 for i in dev if records[i]['batch']==batch)) for batch in sorted({r['batch'] for r in records})}
    successes=int(sum(delta[dev]@direction>0)); observed=successes/36
    strata=defaultdict(list)
    for local,i in enumerate(fit):strata[tuple(records[i]['stratum'])].append(local)
    for key,indices in strata.items():
        indices.sort(key=lambda local:hashlib.sha256(f"BCC-v1|20260912|{records[fit[local]]['batch']}|{records[fit[local]]['train_index']}".encode()).hexdigest())
        require(len(indices)==10,'Incorrect stratum size')
    rng=np.random.Generator(np.random.PCG64(20260912)); exceed=0
    # Dot products and norms via Gram matrices avoid 10k full high-dimensional vectors.
    cross=delta[fit]@delta[dev].T; gram=delta[fit]@delta[fit].T
    for _ in range(permutations):
        s=np.full(40,-1.0)
        for key in sorted(strata):s[rng.choice(strata[key],size=5,replace=False)]=1
        norm2=float(s@gram@s)/1600
        score=float(np.mean(s@cross>0)) if np.isfinite(norm2) and norm2>threshold**2 else 0.0
        exceed+=score>=observed
    pvalue=(1+exceed)/(1+permutations)
    loo=(raw[None,:]*40-delta[fit])/39
    cos=loo@raw/(np.linalg.norm(loo,axis=1)*np.linalg.norm(raw))
    report=dict(status='completed_CPU_development_not_generation',development_positive=successes,
        per_source_positive=counts,permutations=permutations,permutation_seed=20260912,
        permutation_pvalue=pvalue,raw_norm=float(np.linalg.norm(raw)),null_raw_norm=float(np.linalg.norm(null)),
        near_zero_threshold=float(threshold),min_leave_one_out_cosine=float(cos.min()),
        fixed_numeric_gate=successes>=24 and all(x>=11 for x in counts.values()) and pvalue<=.10,
        projections=[dict(pair_id=r['pair_id'],split=r['split'],projection=float(direction@d)) for r,d in zip(records,delta)],
        limitation='Numeric gate only; source/position/termination diagnostics still require review. Not independent efficacy evidence.')
    return report,direction.astype(np.float32),placebo.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Output already exists')
    plan=read(a.prepared/'plan.json');ledger=read(a.replay/'ledger.json')
    require(ledger['status']=='completed' and ledger['plan_sha256']==sha(a.prepared/'plan.json'),'Incomplete or different replay')
    require(ledger['states_sha256']==sha(a.replay/'states.npy'),'States changed')
    require(plan['prefixes_sha256']==sha(a.prepared/'prefixes.json'),'Prefixes changed')
    report,direction,null=diagnostic(read(a.prepared/'prefixes.json'),np.load(a.replay/'states.npy'),plan['parent_vector_norm'])
    a.output.mkdir(parents=True);report.update(parent_fit_sha256=plan['parent_fit_sha256'],replay_ledger_sha256=sha(a.replay/'ledger.json'))
    # NumPy diagnostic assets only; do not masquerade as the old LDA fit metadata.
    np.save(a.output/'bcc_direction.npy',direction);np.save(a.output/'fixed_null_direction.npy',null)
    report['assets_sha256']={p.name:sha(p) for p in a.output.glob('*.npy')};save(a.output/'diagnostic.json',report)
    print(report['fixed_numeric_gate'])


if __name__=='__main__':main()
