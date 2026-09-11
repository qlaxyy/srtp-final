"""Paired question bootstrap intervals; descriptive, never a new promotion gate."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from mechanism_candidates import read, save, sha, require


def analyze(analysis, repetitions=10000, seed=20260912):
    pair=analysis['comparison'];rows=pair['per_question'];n=len(rows)
    groups=list(pair['groups'].values());require(len(groups)==2,'Expected pair')
    correct=np.array([int(r['candidate_correct'])-int(r['original_correct']) for r in rows])
    think=np.array([r['thinking_token_delta'] for r in rows],dtype=float)
    total=np.array([r['total_token_delta'] for r in rows],dtype=float)
    caps=np.array([int(r['candidate_capped'])-int(r['original_capped']) for r in rows])
    require(correct.sum()==groups[1]['correct']-groups[0]['correct'],'Correctness sum')
    require(abs(total.mean()-(groups[1]['mean_total_tokens']-groups[0]['mean_total_tokens']))<1e-8,'Token sum')
    rng=np.random.default_rng(seed);boot={k:[] for k in ('accuracy_pp','thinking_tokens','total_tokens','cap_fraction_pp')}
    for start in range(0,repetitions,250):
        sample=rng.integers(0,n,size=(min(250,repetitions-start),n))
        for key,values,scale in [('accuracy_pp',correct,100),('thinking_tokens',think,1),('total_tokens',total,1),('cap_fraction_pp',caps,100)]:
            boot[key].extend((values[sample].mean(axis=1)*scale).tolist())
    improve=int((correct>0).sum());degrade=int((correct<0).sum());discordant=improve+degrade
    probability=min(1.,2*sum(math.comb(discordant,i) for i in range(min(improve,degrade)+1))/2**discordant) if discordant else 1.
    return dict(count=n,method='Question-paired percentile bootstrap; same resampled question indices in both arms',seed=seed,repetitions=repetitions,
        confidence_level=.95,intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},
        exact_two_sided_mcnemar_p=probability,improved=improve,degraded=degrade,
        limitations=['Conditional on these fixed model runs; does not quantify generation-seed variability.',
                    'No multiplicity correction; descriptive intervals do not change the preregistered gate.',
                    'Small question samples cannot establish a narrow accuracy non-inferiority bound.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Output exists')
    result=analyze(read(a.analysis));result['analysis_sha256']=sha(a.analysis);save(a.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
