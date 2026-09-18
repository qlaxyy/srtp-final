"""Question-cluster bootstrap of calibration class means, no new answers."""
import json
from pathlib import Path
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    c=np.array([s['confidence'] for s in steps]);h=np.array([s['lexical_hit'] for s in steps]);q=np.array([s['question'] for s in steps])
    audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json');length=np.array([r['thinking_tokens'] for r in audit['questions']])
    lo,hi=np.quantile(c,[.25,.75]);over=h|(c<lo);under=~h&(c>hi)
    rng=np.random.default_rng(20260918);weights=rng.multinomial(500,np.full(500,1/500),size=512)
    result={}
    for name,mo,mu in [('original',over,under),('length',over&(length[q]>length.mean()),under&(length[q]<length.mean()))]:
        def aggregate(mask):
            sums=np.zeros((500,x.shape[1]));counts=np.bincount(q[mask],minlength=500)
            for i in range(500):
                ix=mask&(q==i)
                if ix.any():sums[i]=x[ix].sum(0,dtype=np.float64)
            return sums,counts
        so,no=aggregate(mo);su,nu=aggregate(mu)
        original=so.sum(0)/no.sum()-su.sum(0)/nu.sum()
        assert (weights@no>0).all() and (weights@nu>0).all()
        draws=(weights@so)/(weights@no)[:,None]-(weights@su)/(weights@nu)[:,None]
        norms=np.linalg.norm(draws,axis=1);cos=draws@original/(norms*np.linalg.norm(original))
        result[name]=dict(cosine_to_full_vector_quantiles=np.quantile(cos,[.025,.5,.975]).tolist(),
            norm_quantiles=np.quantile(norms,[.025,.5,.975]).tolist(),full_norm=float(np.linalg.norm(original)))
    save(HERE/'length_state_alignment_20260918_run1/stability.json',dict(results=result,replicates=512,seed=20260918,
        method='Resample questions with replacement; retain all selected steps within sampled questions and fixed original thresholds. No state-level independence assumption.',
        limitations='Conditional on current masks/layer and greedy calibration; does not verify semantics or generation efficacy.',
        script_sha256=sha(Path(__file__))))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
