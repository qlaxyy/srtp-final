"""CPU checks against the published p21 score equation and analytic boundaries."""
import argparse
from math import sqrt
from pathlib import Path
from statistics import NormalDist
import numpy as np
from paired_confirmation_stats import score_interval, summarize
from prepare_bcc import save, sha


def reference(b,c,n,original_stopping=False):
    z=NormalDist().inv_cdf(.975);ends=[]
    for sign in (-1,1):
        if (b if sign<0 else c)==n:ends.append(float(sign));continue
        root=(c-b)/n;step=1+root if sign<0 else 1-root
        for _ in range(50 if original_stopping else 70):
            step*=.5;d=root+sign*step
            pb=-b-c+(2*n-c+b)*d;pc=-b*d*(1-d)
            p21=(sqrt(max(0,pb*pb-8*n*pc))-pb)/(4*n)
            sc=(c-b-n*d)/sqrt(n*(2*p21+d*(1-d)))
            if abs(sc)<z:root=d
            if original_stopping and (step<1e-7 or abs(z-sc)<1e-6):break
        ends.append(d)
    return ends


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    fine=[];published=[]
    for c in range(21):
        for b in range(21-c):
            actual=score_interval(c,b,20);reverse=score_interval(b,c,20)
            assert actual[0]<=(c-b)/20<=actual[1]
            assert abs(actual[0]+reverse[1])<1e-12
            fine.append(max(abs(x-y) for x,y in zip(actual,reference(b,c,20))))
            published.append(max(abs(x-y) for x,y in zip(actual,reference(b,c,20,True))))
    assert max(fine)<1e-12
    z=NormalDist().inv_cdf(.975);n=200
    assert abs(score_interval(0,0,n)[1]-z*z/(n+z*z))<1e-12
    assert abs(score_interval(n,0,n)[0]-(n-z*z)/(n+z*z))<1e-12
    x=dict(total=np.full(200,1000),thinking=np.full(200,900),correct=np.ones(200,dtype=int),capped=np.zeros(200,dtype=int))
    y={k:v.copy() for k,v in x.items()};y['total'][:]=800;y['thinking'][:]=700
    positive=summarize(x,y);assert positive['decision']=='independent_confirmation_supported'
    y['correct'][:6]=0;six=summarize(x,y);assert six['decision']=='positive_budget_compatible_replication_not_confirmed'
    y['correct'][6]=0;seven=summarize(x,y);assert seven['decision']=='not_supported_under_fixed_protocol'
    save(a.output,dict(status='CPU_validated_no_generation',
        source='https://raw.githubusercontent.com/cran/PropCIs/master/R/scoreci.mp.R',
        reference='Independent Python evaluation of p21 equation; no R runtime execution. Production uses constrained total-discordance q equation.',
        exhaustive_tables_n20=len(fine),max_tight_reference_difference=max(fine),
        max_original_stopping_reference_difference=max(published),
        first_check_failure='Original published early stopping yields max 2.0019348e-7 endpoint difference, marginally above the initial 2e-7 check. Same equation evaluated to tight convergence agrees <1e-12; production formula and experiment gates unchanged.',
        analytic_boundary_checks=['all concordant nonzero width','all improved endpoint','swap symmetry'],
        synthetic_decisions=[positive['decision'],six['decision'],seven['decision']],
        script_sha256=sha(Path(__file__).with_name('paired_confirmation_stats.py')),new_answers=0))
    print('231 tables, analytic limits, and fixed 3-pp classification checks passed')


if __name__=='__main__':main()
