"""Posthoc CPU common-ascent geometry; never trains or exports a vector.
Minimum-norm point in convex hull: projection optimality implies
<G_i,g*> >= ||g*||^2, hence strict common ascent if g* != 0.
This is a local surrogate fact, not a generation or accuracy guarantee.
"""
import hashlib,itertools,json
from pathlib import Path
import numpy as np

def min_norm_hull(g):
    n=len(g); gram=g@g.T; best=None
    for size in range(1,n+1):
        for ix in itertools.combinations(range(n),size):
            a=gram[np.ix_(ix,ix)]
            kkt=np.block([[a,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(size),1.]
            sol=np.linalg.lstsq(kkt,rhs,rcond=None)[0][:size]
            if abs(sol.sum()-1)>1e-8 or sol.min() < -1e-9:continue
            w=np.zeros(n);w[list(ix)]=sol
            direction=w@g;norm2=float(direction@direction)
            if best is None or norm2<best[0]:best=(norm2,w,direction)
    norm2,w,d=best
    assert (g@d-norm2).min()>-1e-7*max(1,float(np.max(np.sum(g*g,axis=1))))
    return d,w

def main():
    for g in [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[-1.,0.]]),np.array([[1.,1.],[1.,-1.],[2.,0.]])]:
        d,w=min_norm_hull(g)
        assert np.all(g@d>=float(d@d)-1e-9)
    home=Path(__file__).resolve().parent;root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    old=home/'feedback_expanded_20260919_run1'
    out=home/'feedback_theory_20260919_run1';out.mkdir(exist_ok=False)
    base=root/'.codex_work/feedback_expanded_evidence_20260919/results/feedback_expanded_gradient_20260919_run1'
    done=json.loads((base/'complete.json').read_text())
    for r in done['rows']:assert hashlib.sha256((base/r['file']).read_bytes()).hexdigest()==r['sha256']
    parents=json.loads((old/'split.json').read_text())['parents']
    g=np.stack([np.load(base/(str(p['question'])+'_'+p['chosen_source']+'.npz'))['gradient'].astype(float)-np.load(base/(str(p['question'])+'_'+p['rejected_source']+'.npz'))['gradient'].astype(float) for p in parents])
    fit=np.array([p['split']=='fit' for p in parents]);u=np.array([p['chosen_source']=='U' for p in parents]);guard=np.array([p['split']=='correctness_guard_only' for p in parents])
    groups=np.stack([g[fit&u].mean(0),g[fit&~u].mean(0)])
    reports={}
    for name,objectives in [('two_fit_source_means',groups),('two_fit_sources_plus_five_exposed_guards',np.vstack([groups,g[guard]]))]:
        d,w=min_norm_hull(objectives);norm=float(np.linalg.norm(d));unit=d/norm if norm>1e-12 else d
        sets={}
        for role in ['fit','development_check','correctness_guard_only']:
            for source in ['U','L27']:
                mask=np.array([p['split']==role and p['chosen_source']==source for p in parents]);v=g[mask]@unit
                sets[role+':'+source]=dict(n=len(v),positive=int((v>0).sum()),negative=int((v<0).sum()),mean=float(v.mean()),median=float(np.median(v)))
        reports[name]=dict(weights=w.tolist(),direction_norm=norm,objective_projections=(objectives@unit).tolist(),kkt_min_residual=float((objectives@d-norm*norm).min()),groups=sets)
    result=dict(status='Posthoc feasibility diagnosis only, NOT a new validated candidate',theorem='For g*=argmin_{g in convex_hull(G)} ||g||^2, Gi dot g* >= ||g*||^2. Nonzero g* gives common ascent for these finitely many surrogate objectives.',checks='Orthogonal, exactly opposing, and three-objective synthetic KKT checks passed; all archived gradients hashes verified',reports=reports,limits=['Data and developer outcomes were already exposed','Including five guards spends them as constraints; they cease to be validation for that diagnostic','Common group means do not guarantee all individual examples improve','Fixed-history BF16 gradients, unresolved native score outliers','No finite-step likelihood, online-controller, token or correctness guarantee','No vector export, optimizer, GPU, or benchmark'])
    (out/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
