"""Prepare one anchored endpoint candidate on CPU; no fitted-curve claim."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908')
    assert sha(old/'fit.json')=='5bfe9feffc5cc10b494eb881651d863e434520c110e31f8bfa4e1f70778c7a83'
    assert sha(old/'auto_vector.pt')=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'
    z=np.load(H/'results_run3/endpoint_means.npz');v=z['new_O']-z['old_U'];assert v.shape==(1536,) and np.isfinite(v).all()
    base=torch.load(old/'auto_vector.pt',map_location='cpu',weights_only=True)
    assert np.array_equal((z['old_O']-z['old_U']).astype(np.float32),base.numpy())
    out=H/'candidate_new_O_old_U';out.mkdir(exist_ok=False)
    torch.save(torch.from_numpy(v.astype(np.float32)),out/'auto_vector.pt')
    f=json.loads((old/'fit.json').read_text(encoding='utf-8'));params=f['parameters'].copy()
    f.update(version='single-O-update-20260920',method='Diagnostic raw new native O minus frozen old U; full original curve held fixed; NOT refitted or geometrically recalibrated.',
      vector_sha256=sha(out/'auto_vector.pt'),vector_norm=float(np.linalg.norm(v)),
      positives=22038,negatives=11309,calibration_source_sha256='mixed source endpoints; see provenance.json',
      selection={'layer':21,'selection':'fixed common saved layer'},
      endpoint_mean_source_sha256=sha(H/'results_run3/endpoint_means.npz'),curve_source_sha256=sha(old/'fit.json'))
    assert f['parameters']==params
    (out/'fit.json').write_bytes((json.dumps(f,indent=2)+'\n').encode())
    receipt=dict(status='CPU prepared, not evaluated',old_fit_sha256=sha(old/'fit.json'),endpoint_sha256=sha(H/'results_run3/endpoint_means.npz'),
      vector_sha256=sha(out/'auto_vector.pt'),fit_sha256=sha(out/'fit.json'),
      raw_vector_norm=float(np.linalg.norm(v)),old_vector_norm=float(np.linalg.norm(base.numpy())),
      direct_half_step_norm=float(.5*np.linalg.norm(v)),no_norm_matching=True,new_fit=False,new_model_forward_count=0,
      threshold_note='Native O uses lexical OR c<q25; q75/q90 produce identical O. Old U and old runtime high anchor remain q75.',
      limitation='Cross-distribution anchor; old curve retained to isolate endpoint update, not claimed optimal or refitted. Native collection execution differs.')
    (out/'provenance.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
