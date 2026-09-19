"""Reuse original replay states; change high confidence quantile in both uses."""
import hashlib, io, json, sys, tarfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HOME=Path(__file__).resolve().parent
ROOT=next(p for p in HOME.parents if (p/'.git').exists())
HERE=HOME.parent
WORK=ROOT/'.codex_work/base_replay_q90_20260920'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    WORK.mkdir(parents=True,exist_ok=False)
    raw=ROOT/'.codex_work/iterative_recalibration_20260919'
    paths={'calibration_evidence.tar.gz':'a63dbcdce6b40efed1ac35d349f82c3a89b9afae8ba97dce2242936624a72768',
           'selected_features.tar.gz':'97e2281967476b3f854e8a379ab2f05acfc5af1346ef7873d466d268c98085e7'}
    for n,h in paths.items():assert sha(raw/n)==h
    sys.path.insert(0,str(ROOT/'integration/rebalance_easysteer/scripts'))
    import calibrate_auto as cal
    torch.set_num_threads(8)
    report={'scope':'Original base replay states, not native eager collection','archives':paths,'new_answers':0,'model_forwards':0,'groups':{}}
    with tarfile.open(raw/'calibration_evidence.tar.gz') as metadata, tarfile.open(raw/'selected_features.tar.gz') as features:
        for source in ['R','L27']:
            out=WORK/(source+'_fit');out.mkdir()
            prefix=source+'_fit/'
            data={n:json.load(metadata.extractfile(prefix+n+'.json')) for n in ['steps','protocol','selected_layer','fit','collection']}
            steps=data['steps'];layer=data['selected_layer']['best']['layer']
            c=np.array([s['confidence'] for s in steps]);lex=np.array([s['lexical_hit'] for s in steps]);q=np.array([s['question'] for s in steps])
            low,oldhigh=data['protocol']['confidence_quantiles'];high=float(np.quantile(c,.9))
            over=lex|(c<low);oldU=(~lex)&(c>oldhigh);newU=(~lex)&(c>high)
            assert int(oldU.sum())==data['fit']['negatives'] and int(over.sum())==data['fit']['positives']
            assert np.all(~newU|oldU)
            x=np.load(io.BytesIO(features.extractfile(prefix+f'layer_{layer}.npy').read()))
            assert len(x)==len(steps) and np.isfinite(x).all()
            original=torch.load(io.BytesIO(metadata.extractfile(prefix+'auto_vector.pt').read()),map_location='cpu',weights_only=True)
            recomputed=(x[over].mean(0,dtype=np.float64)-x[oldU].mean(0,dtype=np.float64)).astype(np.float32)
            assert np.array_equal(original.numpy(),recomputed),'Original vector not exactly reproduced'
            np.save(out/f'layer_{layer}.npy',x)
            protocol=dict(data['protocol'],confidence_quantiles=[low,high],confidence_quantile_levels=[.25,.9])
            save(out/'protocol.json',protocol);save(out/'steps.json',steps)
            save(out/'selected_layer.json',data['selected_layer'])
            save(out/'collection.json',dict(data['collection'],feature_dir=str(out.resolve())))
            cal.fit(SimpleNamespace(output=out,model=Path(data['fit']['model'])))
            fitted=read(out/'fit.json');assert fitted['parameters']['q75c']==high
            fitted.update(version='base-replay-q90-20260920',model=data['fit']['model'],confidence_high_quantile=.9,
                          legacy_parameter_note='q75c runtime key contains confidence q90; variance q75v unchanged',
                          layer_selection_note='Reuse original automatic layer selection on the identical replay states; no layer search this run')
            (out/'fit.json').write_bytes((json.dumps(fitted,indent=2)+'\n').encode())
            save(out/'complete.json',{'passed':True,'engineering':False,'cpu_only':True,'new_answers':0,'model_forwards':0})
            vnew=torch.load(out/'auto_vector.pt',map_location='cpu',weights_only=True).double();vold=original.double()
            report['groups'][source]=dict(steps=len(steps),over_steps=int(over.sum()),old_under_steps=int(oldU.sum()),new_under_steps=int(newU.sum()),
                old_under_questions=len(np.unique(q[oldU])),new_under_questions=len(np.unique(q[newU])),
                old_high=oldhigh,new_high=high,layer=layer,original_vector_exact=True,
                cosine=float(torch.nn.functional.cosine_similarity(vold,vnew,dim=0)),old_vector_norm=float(original.norm()),new_vector_norm=float(vnew.norm()),
                fit_sha256=sha(out/'fit.json'),vector_sha256=sha(out/'auto_vector.pt'),curve_check=read(out/'curve_check.json'))
    save(WORK/'summary.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
