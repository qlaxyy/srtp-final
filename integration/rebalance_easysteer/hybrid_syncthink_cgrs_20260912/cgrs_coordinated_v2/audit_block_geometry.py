"""Descriptive direction audit only; do not export a deployable vector."""
import json
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    out=HERE/'repeat_content_review_20260918_cpu';ev=read(out/'blocks.json')['events']
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    x=np.load(cache/'layer_21.npy',mmap_mode='r');steps=read(cache/'steps.json')
    old=torch.load(ROOT/'.codex_work/auto_code_v2_500_20260908/auto_vector.pt',weights_only=True).numpy().astype(float)
    c=np.array([s['confidence'] for s in steps]);qs=np.quantile(c,[.25,.75])
    conf=x[c<qs[0]].mean(0,dtype=np.float64)-x[c>qs[1]].mean(0,dtype=np.float64)
    # Compare states at the end of the matched three-step blocks. One pair per question.
    first=np.array([r['first_steps'][-1] for r in ev]);second=np.array([r['repeat_steps'][-1] for r in ev])
    diffs=x[second].astype(float)-x[first].astype(float);d=diffs.mean(0)
    cos=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    rng=np.random.default_rng(20260918);w=rng.multinomial(len(ev),np.full(len(ev),1/len(ev)),size=512)
    boot=w@diffs/len(ev);cs=boot@d/(np.linalg.norm(boot,axis=1)*np.linalg.norm(d))
    dc=c[second]-c[first];positions=np.array([r['repeat_end']-r['first_end'] for r in ev])
    result=dict(questions=len(ev),source_sha256=sha(out/'blocks.json'),features_sha256=sha(cache/'layer_21.npy'),
        raw_norm=float(np.linalg.norm(d)),old_norm=float(np.linalg.norm(old)),cosine_original=cos(d,old),
        cosine_low_minus_high_confidence=cos(d,conf),confidence_delta_mean=float(dc.mean()),
        absolute_confidence_delta_quantiles=np.quantile(abs(dc),[.25,.5,.75,.9]).tolist(),
        position_gap_quantiles=np.quantile(positions,[0,.5,.9,1]).tolist(),
        question_bootstrap_cosine_quantiles=np.quantile(cs,[.025,.5,.975]).tolist(),
        repeated_end_low_confidence_questions=int(sum(c[second]<qs[0])),
        repeated_end_high_confidence_questions=int(sum(c[second]>qs[1])),
        note='End of third step; all 145 pairs equally weighted. No scaling, vector export, fit, GPU or causal claim. Positive repeat-minus-first aligns with higher confidence if cosine to low-minus-high is negative. Stability does not establish semantic specificity.')
    save(out/'geometry.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
