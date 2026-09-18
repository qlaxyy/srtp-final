"""Freeze the user's length-conditioned class-mean vector, retaining old controller."""
import hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('E:/srtp/srtp-final')

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    rel='trajectory_length_vector_20260918_run2';out=HERE/rel;out.mkdir(exist_ok=False)
    audit_path=HERE/'trajectory_length_labels_20260918_run1/result.json';audit=read(audit_path)
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');frozen=ROOT/'.codex_work/auto_code_v2_500_20260908'
    assert sha(cache/'steps.json')==audit['input_sha256']['steps']
    x=np.load(cache/'layer_21.npy',mmap_mode='r');assert x.shape==(84008,1536)
    c=np.array([s['confidence'] for s in steps]);q=np.array([s['question'] for s in steps])
    h=np.array([s['lexical_hit'] for s in steps],dtype=bool)
    low,high=read(frozen/'protocol.json')['confidence_quantiles']
    over=h|(c<low);under=~h&(c>high)
    lengths=np.array([r['thinking_tokens'] for r in audit['questions']]);mean=lengths.mean()
    mo=over&(lengths[q]>mean);mu=under&(lengths[q]<mean)
    assert mo.sum()==39515 and mu.sum()==854 and not (mo&mu).any()
    def center(mask):return np.mean(x[mask].astype(np.float64),axis=0)
    old=center(over)-center(under);new=center(mo)-center(mu)
    original=torch.load(frozen/'auto_vector.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.from_numpy(old.astype(np.float32)),original)
    assert np.isfinite(new).all() and np.linalg.norm(new)>0
    vector=torch.from_numpy(new.astype(np.float32));torch.save(vector,out/'auto_vector.pt')
    norm_vector=vector*(original.double().norm()/vector.double().norm()).to(vector.dtype)
    assert abs(float(norm_vector.double().norm()/original.double().norm())-1)<1e-6
    torch.save(norm_vector,out/'norm_vector.pt')
    def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    # Descriptive in-sample geometry only. No fitting to benchmark outcomes.
    q_features=np.array([x[q==i].mean(0,dtype=np.float64) for i in range(500)])
    q_conf=np.array([c[q==i].mean() for i in range(500)])
    def correlations(v):
        z=q_features@v/np.linalg.norm(v)
        design=np.column_stack([np.ones(500),q_conf])
        residual_z=z-design@np.linalg.lstsq(design,z,rcond=None)[0]
        loglen=np.log1p(lengths)
        residual_l=loglen-design@np.linalg.lstsq(design,loglen,rcond=None)[0]
        return dict(question_projection_vs_mean_confidence=float(np.corrcoef(z,q_conf)[0,1]),
            question_projection_vs_log_length=float(np.corrcoef(z,loglen)[0,1]),
            projection_vs_log_length_after_linear_confidence_adjustment=float(np.corrcoef(residual_z,residual_l)[0,1]))
    checks=dict(original_vector_exact=True,over=int(mo.sum()),under=int(mu.sum()),mean_thinking_tokens=float(mean),
        old_norm=float(np.linalg.norm(old)),new_norm=float(np.linalg.norm(new)),cosine=cos(old,new),
        angle_degrees=float(np.degrees(np.arccos(np.clip(cos(old,new),-1,1)))),
        old_projection=correlations(old),new_projection=correlations(new),
        limitations='In-sample descriptive geometry, linear adjustment is not causal disentanglement; truncated length and question difficulty remain confounded.',
        assets=dict(features_sha256=sha(cache/'layer_21.npy'),audit_sha256=sha(audit_path),vector_sha256=sha(out/'auto_vector.pt')))
    save(out/'cpu_checks.json',checks)
    parent=HERE/'vcm_signal_audit_20260918/margin_run1'
    previous=read(parent/'plan.json');oldrelease=read(parent/'release.json')
    plan={k:previous[k] for k in ('rows','model','dataset','seed','max_new_tokens','temperature','top_p','execution','decision','reference_files')}
    plan.update(experiment_kind='length_vector_v1',run_id=rel,arms=[dict(name='LENGTH_L27',suppression_table=rel+'/opening.npz',vector=rel+'/auto_vector.pt'),dict(name='LENGTH_NORM_L27',suppression_table=rel+'/opening.npz',vector=rel+'/norm_vector.pt')],
        new_answers=1000,hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=3600,
        change='Only raw steering-vector class membership changes: original OR over label AND trajectory longer than 500 mean; original AND under label AND shorter. Keep original layer, curve parameters, magnitude conventions and L27 suppression.',
        stored_direction='mu_over-mu_under; negative ReBalance coefficient moves toward under prototype; no sign reversal',
        interpretation='Exposed full benchmark exploration, not independent confirmation; confidence-adjusted CPU correlations are descriptive only.',
        engineering=dict(rows=8,arms=['L27_REFERENCE','LENGTH_L27','LENGTH_REPEAT','LENGTH_NORM_L27','LENGTH_NORM_REPEAT'],max_tokens=512,
            gate='exact original vector reproduction on CPU; unchanged runtime hashes; candidate repeat exact tokens/control histories after reset; finite runtime and complete outputs'),
        advancement='Relative L27: both mean thinking and total tokens decrease, caps do not increase, accuracy loss <=2pp versus both L27 and R. Otherwise no GSM transfer or post-hoc threshold tuning.',
        expected_cost='1.5B MATH500 two distinct arms; 8 old engineering prompts x5 x512 first; roughly15-25 minutes, generation limit1800s/arm, process3600s.',
        secondary_arm='Same new direction rescaled to original vector norm: isolates direction effect from increased magnitude. Predeclared before any new GPU output. Both arms reported; no threshold search.',
        supersedes='run1 CPU-only packaging; never launched. run2 adds norm-matched diagnostic before GPU testing.')
    save(out/'plan.json',plan)
    for n in ('opening.npz','historical_compact.json'):(out/n).write_bytes((parent/n).read_bytes())
    src=list(oldrelease['source_sha256'])+['prepare_length_vector.py']
    release={k:oldrelease[k] for k in ('table_schema','assets','engineering_rows','runtime_root','runtime_source_sha256','source_hash_mode')}
    release.update(status='CPU verified; native engineering required before full500',plan_relative_path=rel+'/plan.json',
        plan_sha256=sha(out/'plan.json'),artifact_root=rel,
        artifact_sha256={n:sha(out/n) for n in ('auto_vector.pt','norm_vector.pt','opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in src})
    save(out/'release.json',release)
    archive=HERE.parents[3]/('.codex_work/'+rel+'.tar.gz')
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in src}
    for n in ('release.json','plan.json','opening.npz','historical_compact.json','cpu_checks.json','auto_vector.pt','norm_vector.pt'):files[rel+'/'+n]=(out/n).read_bytes()
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tf:
            for name,data in files.items():
                info=tarfile.TarInfo(name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
    print(json.dumps(dict(checks=checks,archive=str(archive),archive_sha256=sha(archive)),indent=2))

if __name__=='__main__':main()
