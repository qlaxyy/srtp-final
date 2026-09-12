"""Paired score inversion for M-O accuracy and paired token bootstrap.

Score variance uses the constrained trinomial MLE for probabilities of
improvement, degradation and concordance. Reference: Tango (1998);
CRAN PropCIs scoreci.mp (b=degraded, c=improved, n=all pairs).
https://search.r-project.org/CRAN/refmans/PropCIs/html/scoreci.mp.html
"""
from math import sqrt
from statistics import NormalDist
import numpy as np


def score_interval(improved, degraded, n, confidence=.95):
    if not all(isinstance(v,int) for v in (improved,degraded,n)) or n<=0 or min(improved,degraded)<0 or improved+degraded>n:
        raise ValueError('Invalid paired counts')
    if not 0<confidence<1: raise ValueError('Invalid confidence')
    estimate=(improved-degraded)/n; z=NormalDist().inv_cdf((1+confidence)/2)
    def accepted(delta):
        # q = P(improve)+P(degrade), constrained to |delta| <= q <= 1.
        discordant=improved+degraded; signed=improved-degraded
        linear=discordant+signed*delta
        constant=-signed*delta+(n-discordant)*delta*delta
        q=(linear+sqrt(max(0.,linear*linear+4*n*constant)))/(2*n)
        q=min(1.,max(abs(delta),q))
        variance=q-delta*delta
        return n*(estimate-delta)**2 <= z*z*max(variance,0.)
    def edge(outside,inside):
        if accepted(outside):return outside
        for _ in range(70):
            middle=(outside+inside)/2
            if accepted(middle):inside=middle
            else:outside=middle
        return (outside+inside)/2
    return [edge(-1.,estimate),edge(1.,estimate)]


def summarize(control, candidate, accuracy_margin=.03):
    """Arrays: total/thinking/correct/capped, one item per paired question."""
    x={k:np.asarray(v) for k,v in control.items()};y={k:np.asarray(v) for k,v in candidate.items()}
    n=len(x['total'])
    if n!=200 or any(len(v)!=n for v in list(x.values())+list(y.values())):raise ValueError('Require complete 200 pairs')
    if accuracy_margin!=.03:raise ValueError('Fixed prospective margin is 3 percentage points')
    if any(not np.isfinite(v).all() for v in list(x.values())+list(y.values())):raise ValueError('Nonfinite data')
    for g in (x,y):
        if np.any(g['total']<=0) or np.any(g['total']>16000) or np.any(g['thinking']<0) or np.any(g['thinking']>g['total']):raise ValueError('Token budget/count')
        if any(not np.isin(g[k],[0,1]).all() for k in ('correct','capped')):raise ValueError('Nonbinary outcome')
    rng=np.random.Generator(np.random.PCG64(20260912));boot={k:[] for k in ('total','thinking')}
    for _ in range(80):
        sample=rng.integers(0,n,(250,n))
        for k in boot:
            den=x[k][sample].sum(1)
            if np.any(den<=0):raise ValueError('Undefined token ratio')
            boot[k].extend((y[k][sample].sum(1)/den-1).tolist())
    metrics={k:dict(relative_change=float(y[k].sum()/x[k].sum()-1),
                   ci95=np.quantile(boot[k],[.025,.975]).tolist()) for k in boot}
    difference=y['correct'].astype(int)-x['correct'].astype(int)
    improved=int((difference>0).sum());degraded=int((difference<0).sum())
    accuracy=float(difference.mean());interval=score_interval(improved,degraded,n)
    # Compare integer counts to avoid floating ambiguity at exactly -3 pp.
    point_ok=all(m['relative_change']<0 for m in metrics.values()) and 100*(improved-degraded)>=-3*n and y['capped'].sum()<=x['capped'].sum()
    strict=point_ok and metrics['total']['ci95'][1]<0 and interval[0]>-.03
    return dict(count=n,token_metrics=metrics,accuracy_difference=accuracy,accuracy_score_ci95=interval,
        improved=improved,degraded=degraded,control_caps=int(x['capped'].sum()),candidate_caps=int(y['capped'].sum()),
        accuracy_margin=.03,bootstrap_repetitions=20000,seed=20260912,
        decision='independent_confirmation_supported' if strict else 'positive_budget_compatible_replication_not_confirmed' if point_ok else 'not_supported_under_fixed_protocol',
        limitation='One fixed candidate and generation seed. No automatic expansion, no pooled old screen; score interval is asymptotic.')
