"""CPU feasibility only: same-question, same-text first/repeat state pairs."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'matched_repeat_pairs_20260918_cpu';out.mkdir(exist_ok=False)
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive) as t:rows=[json.loads(s) for s in t.extractfile('generations.jsonl').read().splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json')
    lens=np.array([r['thinking_tokens'] for r in audit['questions']]);mean=lens.mean()
    c=np.array([s['confidence'] for s in steps]);quartiles=np.quantile(c,[.25,.5,.75]);bins=np.searchsorted(quartiles,c,side='right')
    seen=[{} for _ in rows];used=[set() for _ in rows];pairs=[]
    for i,s in enumerate(steps):
        q=s['question']
        if lens[q]<=mean:continue
        ids=rows[q]['token_ids'][s['start']:s['stop']]
        text=' '.join(dec.decode(ids).lower().split())
        if len(ids)<16:continue
        if text not in seen[q]:seen[q][text]=i;continue
        if text in used[q]:continue
        first=seen[q][text]
        # Fix one pair per unique repeated text: first occurrence and first
        # recurrence. Do not search later recurrences for a favorable match.
        used[q].add(text)
        pairs.append(dict(question=q,train_index=rows[q]['train_index'],problem_sha256=audit['questions'][q]['problem_sha256'],
            first_step=first,repeat_step=i,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            first_confidence=float(c[first]),repeat_confidence=float(c[i]),
            same_confidence_quartile=bool(bins[first]==bins[i]),
            first_token_count=steps[first]['stop']-steps[first]['start'],repeat_token_count=len(ids)))
    orig=torch.load(ROOT/'.codex_work/auto_code_v2_500_20260908/auto_vector.pt',weights_only=True).numpy().astype(np.float64)
    pure=x[c<quartiles[0]].mean(0,dtype=np.float64)-x[c>quartiles[2]].mean(0,dtype=np.float64)
    def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    results={}
    for name,ps in [('all_first_repeat_pairs',pairs),('same_quartile',[p for p in pairs if p['same_confidence_quartile']])]:
        if not ps:
            results[name]=dict(pairs=0,questions=0);continue
        diffs=np.array([x[p['repeat_step']].astype(np.float64)-x[p['first_step']] for p in ps]);qs=np.array([p['question'] for p in ps])
        unique=np.unique(qs);means=np.array([diffs[qs==q].mean(0) for q in unique]);direction=means.mean(0)
        delta=np.array([p['repeat_confidence']-p['first_confidence'] for p in ps])
        # Equal question weight, then equal unique-text weight within question.
        rng=np.random.default_rng(20260918);weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=512)
        draws=(weights@means)/len(unique);norms=np.linalg.norm(draws,axis=1)
        cosines=draws@direction/(norms*np.linalg.norm(direction))
        norms_ratio=float(np.linalg.norm(orig)/np.linalg.norm(direction))
        vector=torch.from_numpy((direction*norms_ratio).astype(np.float32));torch.save(vector,out/(name+'.pt'))
        results[name]=dict(pairs=len(ps),questions=len(unique),raw_norm=float(np.linalg.norm(direction)),
            cosine_original=cos(direction,orig),cosine_confidence_only=cos(direction,pure),
            mean_confidence_delta=float(delta.mean()),median_abs_confidence_delta=float(np.median(np.abs(delta))),
            question_balanced_confidence_delta=float(np.mean([delta[qs==q].mean() for q in unique])),
            bootstrap_cosine_quantiles=np.quantile(cosines,[.025,.5,.975]).tolist(),
            vector_sha256=sha(out/(name+'.pt')))
    result=dict(status='CPU feasibility; neither vector is promoted or GPU-tested',
        protocol='Original long trajectories only (>6307.14 thinking tokens); normalized exact step >=16 tokens; first recurrence only; confidence-matched variant uses same original global quartile; equal question then unique-text weights; norm matched to old vector.',
        results=results,pairs=pairs,confidence_quartiles=quartiles.tolist(),
        limitations=['Same text does not guarantee first occurrence is valid or recurrence unnecessary.',
            'Repeated state occurs later; position remains confounded. Quartile match does not equalize exact confidence.',
            'No new answers or hidden-state collection; no use of BCC assets, fitting or splits.'],
        inputs=dict(calibration_source_sha256=audit['source_sha256'],steps_sha256=sha(cache/'steps.json'),features_sha256=sha(cache/'layer_21.npy'),script_sha256=sha(Path(__file__))))
    save(out/'result.json',result);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
