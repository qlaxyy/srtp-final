"""Supplementary CPU attribution checks, not a causal or performance test."""
import io
import json
import hashlib
import tarfile
from pathlib import Path
import numpy as np
from diagnose_paired_endpoints import cosine


def main():
    home=Path(__file__).resolve().parent
    root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    out=home/'paired_endpoint_diagnostic_20260918_run1'
    t=tarfile.open(root/'.codex_work/outcome_endpoints_20260918_evidence.tar.gz')
    prefix='outcome_endpoints_20260918_run1/features/'
    manifest=json.load(t.extractfile(prefix+'manifest.json')); data={}; axes={}
    for r in manifest:
        b=t.extractfile(prefix+r['file']).read();assert hashlib.sha256(b).hexdigest()==r['sha256']
        x=np.load(io.BytesIO(b))['features'].astype(np.float64)
        data[r['question'],r['side']]=(r,x)
        if r['kind']=='CC':
            c=np.array([s['confidence'] for s in r['steps']]); lo,hi=np.quantile(c,[.25,.75])
            if lo<hi:
                axes.setdefault(r['question'],[]).append(x[c<=lo].mean(0)-x[c>=hi].mean(0))
    axis=np.mean([np.mean(v,axis=0) for v in axes.values()],axis=0)
    selections=json.loads((out/'question_selection.json').read_text());result={}
    for name,rows in selections.items():
        x=np.stack([data[r['question'],'long'][1][r['long_step_indices']].mean(0)-data[r['question'],'short'][1][r['short_step_indices']].mean(0) for r in rows])
        groups={s:x[[r['long_source']==s for r in rows]] for s in ['U','L27']}
        balanced=(groups['U'].mean(0)+groups['L27'].mean(0))/2
        splits=[];rng=np.random.default_rng(20260918)
        for _ in range(200):
            aa=[];bb=[]
            for g in groups.values():
                ix=rng.permutation(len(g));h=len(g)//2
                aa.append(g[ix[:h]].mean(0));bb.append(g[ix[h:]].mean(0))
            splits.append(cosine(np.mean(aa,0),np.mean(bb,0)))
        result[name]=dict(cosine_with_within_trajectory_low_minus_high_axis=cosine(x.mean(0),axis),
            source_equal_mean_cosine_with_original=cosine(balanced,x.mean(0)),
            source_equal_norm=float(np.linalg.norm(balanced)),
            source_equal_split_cos_quantiles=np.quantile(splits,[0,.05,.5,.95,1]).tolist())
    # Hold all O states fixed; remove each short-error parent from U only.
    fit=json.loads((home/'outcome_endpoints_20260918_run1/fitted_audited/UNDER_REFIT/report.json').read_text())
    cl,ch=fit['quantiles']['confidence'];o=[];u={}
    for (q,side),(r,x) in data.items():
        c=np.array([s['confidence'] for s in r['steps']]);lex=np.array([s['lexical_hit'] for s in r['steps']])
        if r['kind']=='CC' and side=='long':o.append(x[lex|(c<cl)])
        if q in fit['other_parents'] and side=='short':u[q]=x[(~lex)&(c>ch)]
    mo=np.concatenate(o).mean(0);mu=np.concatenate(list(u.values())).mean(0);d=mo-mu
    result['under_refit']=dict(raw_norm=float(np.linalg.norm(d)),
        short_parent_counts={str(q):len(v) for q,v in u.items()},
        equal_parent_cosine=cosine(d,mo-np.mean([v.mean(0) for v in u.values()],0)),
        remove_parent=[dict(question=q,cosine=cosine(d,mo-np.concatenate([v for p,v in u.items() if p!=q]).mean(0))) for q in u])
    result['limits']=['Within-trajectory confidence axis is descriptive and shares training data.',
        'Source-balanced checks are diagnostic, not a new selected deployment recipe.',
        'No semantic step labels or causal intervention effects are inferred.']
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (out/'supplement.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
