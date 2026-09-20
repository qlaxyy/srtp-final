"""Calibration-only feasibility audit; never reads evaluation outcomes."""
import argparse, hashlib, json, tarfile
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--main-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tarfile.open(a.main_root/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:
        raw=t.extractfile('generations.jsonl').read()
    digest=hashlib.sha256(raw).hexdigest()
    assert digest=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    rows=[json.loads(x) for x in raw.splitlines()]
    steps=json.loads((a.main_root/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json').read_text())
    probs=[np.exp(np.asarray(r['logprobs'],dtype=np.float64)) for r in rows]
    arithmetic=[];harmonic=[];variances=[];previous={}
    for s in steps:
        q=s['question'];ps=probs[q][s['start']:s['stop']]
        assert len(ps)>0 and np.all(np.isfinite(ps)) and np.all((ps>0)&(ps<=1))
        c=float(ps.mean());assert abs(c-s['confidence'])<1e-12
        h=float(len(ps)/np.sum(1/ps));assert h<=c+1e-12
        arithmetic.append(c);harmonic.append(h)
        variances.append((h-previous[q])**2/4 if q in previous else 0.)
        previous[q]=h
    c=np.asarray(arithmetic);h=np.asarray(harmonic);v=np.asarray(variances)
    oldq=np.quantile(c,[.25,.75]);newq=np.quantile(h,[.25,.75]);vq=np.quantile(v,[.25,.75])
    oldbins=np.digitize(c,oldq);newbins=np.digitize(h,newq)
    confusion=np.zeros((3,3),dtype=int);np.add.at(confusion,(oldbins,newbins),1)
    fit=json.loads((a.main_root/'.codex_work/auto_code_v2_500_20260908/fit.json').read_text())
    params=dict(fit['parameters']);params.update(q25c=float(newq[0]),q75c=float(newq[1]),q25v=float(vq[0]),q75v=float(vq[1]))
    params['curve_tau']=min(.01,.5*(-params['low_val_1'])*(1-newq[1])/(newq[1]-newq[0]))
    report=dict(status='CPU feasibility only; no efficacy evidence',questions=len(rows),steps=len(steps),source_sha256=digest,
        original_quantiles=oldq.tolist(),harmonic_parameters=params,
        mean_confidence_drop=float((c-h).mean()),confidence_drop_quantiles=np.quantile(c-h,[0,.25,.5,.75,.95,1]).tolist(),
        changed_confidence_bin_count=int((oldbins!=newbins).sum()),confusion_old_rows_new_columns=confusion.tolist(),
        formula='n/sum(1/p); p is raw maximum probability, not ATAR generated-token probability',
        vector_policy='Keep frozen vector, layer and coefficient amplitudes; recalibrate four signal quantiles and feasible tau only',
        limits='Greedy calibration probabilities reconstruct original arithmetic means; not generated outcomes. No semantic correctness labels.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
