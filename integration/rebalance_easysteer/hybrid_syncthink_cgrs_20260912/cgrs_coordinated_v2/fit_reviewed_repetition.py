"""Fit and gate one reviewed within-trajectory repetition direction on CPU."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import torch


def main():
    home=Path(__file__).resolve().parent
    root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    out=home/'reviewed_repetition_20260918_run1';out.mkdir(exist_ok=False)
    accepted=[8,18,21,48,71,87,127,176,179,204,211,217,219,221,250,257,279,304,313,329,338,340,367,401,413,418,431,449,453,456,462,469,483]
    protocol=dict(status='Fixed before vector geometry or new benchmark generation',questions=accepted,
        direction='Equal question mean of later repeated correct-operation first-token state minus earlier same-operation first-token state, within the same long trajectory.',
        labels='Assistant audited mathematical operation and neighboring repeat context; not independent annotations or causal dispensability labels. Earlier occurrence may already be a check; not assumed globally first or efficient.',
        review='All123 top retrieval candidates screened;38 shortlisted with repeat neighbors inspected; defer132/156/168 for correction or ambiguity,432/447 for context dependency. Keep33.',
        fixed_gates=dict(min_parents=30,min_parents_per_source=5,min_loo_cos=.95,min_random_half_median_cos=.5,min_source_cos=0),
        weighting='Exactly1 state difference per parent; no confidence threshold; no post-hoc removal using geometry.',
        normalization='Match frozen original ReBalance vector norm only if all CPU gates pass.',
        future_eval='If gates pass, candidate-only1.5B MATH500, original controller/layer/L27 unchanged, historical L27/R, seed42 cap16000. No GPU evaluation if any gate fails.',
        limitations=['Same-trajectory earlier/later positions remain a confound.','Negative coefficient moves toward earlier computation, which may induce recomputation; performance is empirical.','Local valid arithmetic does not prove whole trajectory valid.','MATH500 exposed; no independent confirmation.'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n')
    pool=json.loads((root/'.codex_work/repetition_review_pool.json').read_text(encoding='utf8'))
    chosen=[r for r in pool if r['question'] in accepted];assert len(chosen)==len(accepted)==33
    archive=root/'.codex_work/outcome_endpoints_20260918_evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='ba391343b8b86e28514f4dc638a7bc189979883727a17b3fe0f1b757bfce22ff'
    t=tarfile.open(archive);pre='outcome_endpoints_20260918_run1/features/'
    m={(r['question'],r['side']):r for r in json.load(t.extractfile(pre+'manifest.json'))}
    differences=[];rows=[]
    for r in chosen:
        q=r['question'];mm=m[q,'long'];b=t.extractfile(pre+mm['file']).read();assert hashlib.sha256(b).hexdigest()==mm['sha256']
        x=np.load(io.BytesIO(b))['features'].astype(np.float64);first=r['first_step'];repeat=r['repeat_step'];assert first<repeat
        differences.append(x[repeat]-x[first]);rows.append(dict(r,earlier_step_metadata=mm['steps'][first],repeat_step_metadata=mm['steps'][repeat],state_source_sha256=mm['sha256']))
    x=np.stack(differences);d=x.mean(0);assert np.isfinite(x).all()
    def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    loo=[cos(d,(x.sum(0)-v)/(len(x)-1)) for v in x];splits=[];rng=np.random.default_rng(20260918)
    for _ in range(200):
        ix=rng.permutation(len(x));h=len(x)//2;splits.append(cos(x[ix[:h]].mean(0),x[ix[h:]].mean(0)))
    groups={s:x[[r['long_source']==s for r in rows]] for s in ['U','L27']}
    sourcecos=cos(groups['U'].mean(0),groups['L27'].mean(0))
    gates=dict(parents=len(rows)>=30,source_counts=min(len(v) for v in groups.values())>=5,loo=min(loo)>=.95,split_median=np.median(splits)>=.5,source_cos=sourcecos>=0)
    originalpath=Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908/auto_vector.pt')
    assert hashlib.sha256(originalpath.read_bytes()).hexdigest()=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'
    original=torch.load(originalpath,map_location='cpu',weights_only=True).double().numpy()
    report=dict(status='passed CPU readiness only' if all(gates.values()) else 'failed CPU readiness; no GPU run',gates={k:bool(v) for k,v in gates.items()},
        raw_norm=float(np.linalg.norm(d)),original_norm=float(np.linalg.norm(original)),cosine_with_original=cos(d,original),
        source_counts={s:len(v) for s,v in groups.items()},source_cos=sourcecos,loo_min=min(loo),loo_worst_question=rows[int(np.argmin(loo))]['question'],
        split_cos_quantiles=np.quantile(splits,[0,.05,.5,.95,1]).tolist(),
        earlier_repeat_confidence=[float(np.mean([r[k]['confidence'] for r in rows])) for k in ['earlier_step_metadata','repeat_step_metadata']],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(out/'diagnostic_differences.npz',questions=np.array(accepted),differences=x,mean_difference=d)
    if all(gates.values()):
        v=torch.from_numpy((d*np.linalg.norm(original)/np.linalg.norm(d)).astype(np.float32));torch.save(v,out/'auto_vector.pt')
        assert abs(float(v.double().norm())/np.linalg.norm(original)-1)<1e-6
        report['vector_sha256']=hashlib.sha256((out/'auto_vector.pt').read_bytes()).hexdigest()
    for name,obj in [('report',report),('reviewed_pairs',rows)]:
        (out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
