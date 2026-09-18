"""Precommitted source-stratified first-order diagnostics, no fitted vector."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',type=Path,required=True);ap.add_argument('--gradients',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    split=json.loads(a.split.read_text());done=json.loads((a.gradients/'complete.json').read_text())
    assert done['optimizer_steps']==0 and done['backward_count']==452
    grads={};lengths={}
    for r in done['rows']:
        assert sha(a.gradients/r['file'])==r['sha256']
        grads[r['key']]=np.load(a.gradients/r['file'])['gradient'].astype(np.float64)
        lengths[r['key']]=r['response_tokens']
    parents=split['parents'];diffs=[];normalized=[]
    for p in parents:
        c=str(p['question'])+'_'+p['chosen_source'];r=str(p['question'])+'_'+p['rejected_source']
        diffs.append(grads[c]-grads[r]);normalized.append(grads[c]/lengths[c]-grads[r]/lengths[r])
    d=np.stack(diffs);fit=np.array([p['split']=='fit' for p in parents]);u=np.array([p['chosen_source']=='U' for p in parents]);l=~u
    mean=d[fit].mean(0);unit=mean/np.linalg.norm(mean)
    balanced=(d[fit&u].mean(0)+d[fit&l].mean(0))/2
    groups={};rng=np.random.default_rng(20260919)
    for role in ['fit','development_check','correctness_guard_only']:
        for source in ['U','L27']:
            mask=np.array([p['split']==role and p['chosen_source']==source for p in parents]);x=d[mask]@unit
            boots=np.mean(x[rng.integers(0,len(x),size=(20000,len(x)))],axis=1)
            groups[role+':'+source]=dict(n=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()),mean_projection=float(x.mean()),median_projection=float(np.median(x)),mean_projection_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),cosine_mean_to_fit=cosine(d[mask].mean(0),mean))
    guards=np.array([p['split']=='correctness_guard_only' for p in parents])
    passed=all(groups['development_check:'+s]['median_projection']>0 for s in ['U','L27']) and bool(np.all(d[guards]@unit>=0))
    nd=np.stack(normalized)
    report=dict(status='First-order gradient diagnosis, no optimized vector or efficacy result',decision='eligible_for_separate_training_plan' if passed else 'do_not_advance_pooled_direction',source_cosine_fit=cosine(d[fit&u].mean(0),d[fit&l].mean(0)),source_cosine_length_normalized_diagnostic=cosine(nd[fit&u].mean(0),nd[fit&l].mean(0)),groups=groups,parents=[dict(question=p['question'],split=p['split'],chosen_source=p['chosen_source'],projection=float(d[i]@unit),balanced_projection_diagnostic=float(d[i]@balanced/np.linalg.norm(balanced))) for i,p in enumerate(parents)],forward_count=done['model_forward_count'],backward_count=done['backward_count'],wall_seconds=done['wall_seconds'],compute_seconds=sum(r['seconds'] for r in done['rows']),peak_GiB=max(r['peak_allocated_GiB'] for r in done['rows']),input_sha256=dict(split=sha(a.split),complete=sha(a.gradients/'complete.json')),limits=['Fixed-history BF16 first-order surrogate, not actual nonlinear or online-policy change','Bootstrap conditional on fitted direction, excludes fit/model/schedule uncertainty','Already exposed training data, not independent confirmation','Five guards cannot establish accuracy noninferiority','Balanced and length-normalized directions are diagnostics only, not fallback candidates'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['parents','limits','input_sha256']},indent=2))
if __name__=='__main__':main()
