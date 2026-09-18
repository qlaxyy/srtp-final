"""CPU-only geometry audit; no new model outputs or online effect estimates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('E:/srtp/srtp-final')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def summary(a):
    return dict(zip(('min','p05','median','p95','max'),
                    map(float, np.quantile(a, [0,.05,.5,.95,1]))))

def main():
    out = HERE / 'norm_preservation_20260918_cpu'
    out.mkdir(exist_ok=False)
    features = ROOT / '.codex_work/question_balanced_20260911/original_selected_layer/layer_21.npy'
    vector = ROOT / '.codex_work/auto_code_v2_500_20260908/auto_vector.pt'
    expected = {features: '66fdc33b5ca4d40d76243c1ff062dabdecbe542bcc02580ee0cb1ae2ea31ba66',
                vector: 'fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'}
    for p, digest in expected.items():
        assert sha(p) == digest, str(p)
    protocol = dict(source='https://arxiv.org/html/2601.19375v1',
        sentinel='END-OF-PAPER:3cef11e77ca3',
        question='Does original additive steering change complete-state norm on frozen calibration states?',
        alphas=[-1.5353450143175593,-1.0,0.1],
        selection='All 84008 original step-start features; all 500 old calibration parents; no new split or reserved problems.',
        proposed_map='y=x+alpha*v; z=y*norm(x)/norm(y); zero norm y falls back to x; alpha zero and disabled branch return old path exactly.',
        scope='Fixed-coefficient stress audit, NOT actual online coefficient frequency or efficacy. Unsteered first-content-token complete states only.',
        inputs={str(p): d for p,d in expected.items()})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    x=np.load(features,mmap_mode='r')
    v=torch.load(vector,map_location='cpu',weights_only=True).double().numpy()
    assert x.shape==(84008,1536)
    n2=np.empty(len(x)); dot=np.empty(len(x))
    for start in range(0,len(x),1024):
        a=x[start:start+1024].astype(np.float64)
        n2[start:start+len(a)]=np.einsum('ij,ij->i',a,a)
        dot[start:start+len(a)]=a@v
    vv=v@v
    rows=[]
    for alpha in protocol['alphas']:
        yn2=n2+2*alpha*dot+alpha*alpha*vv
        assert (yn2>0).all()
        ratio=np.sqrt(yn2/n2)
        cosine=(n2+alpha*dot)/np.sqrt(n2*yn2)
        rows.append(dict(alpha=alpha,norm_ratio=summary(ratio),
            angle_degrees=summary(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
            fraction_abs_norm_change_above_1pct=float(np.mean(abs(ratio-1)>.01)),
            fraction_abs_norm_change_above_5pct=float(np.mean(abs(ratio-1)>.05)),
            renormalization_scale=summary(1/ratio)))
    # Counterexample to Appendix B.1 Eq18, under the paper's orthonormal assumption.
    b=np.eye(2); h=np.array([0.,1.]); theta=0.
    r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    hp=h-b@b.T@h+np.linalg.norm(b@b.T@h)*b@r@np.array([1.,0.])
    assert not np.array_equal(h,hp) and np.linalg.norm(h)==np.linalg.norm(hp)
    # Verify rescaling on representative rows and complete-state split identity.
    a=x[::101].astype(np.float64); y=a-v; z=y*(np.linalg.norm(a,axis=1)/np.linalg.norm(y,axis=1))[:,None]
    err=float(np.max(abs(np.linalg.norm(z,axis=1)/np.linalg.norm(a,axis=1)-1)))
    assert err<1e-12
    residual=.3*a; hidden=a-residual
    assert np.allclose((hidden+(z-a))+residual,z,rtol=1e-12,atol=1e-12)
    result=dict(states=len(x),vector_norm=float(np.sqrt(vv)),state_norm=summary(np.sqrt(n2)),
        rows=rows,paper_eq18_counterexample=dict(h=h.tolist(),hprime=hp.tolist(),both_norms=1),
        rescale_max_relative_norm_error=err,split_residual_identity_passed=True,
        decision='CPU geometry only; do not claim reasoning compression, improved accuracy, or actual online norm distribution. Runtime integration and GPU validation remain undone.',
        script_sha256=sha(Path(__file__)))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
