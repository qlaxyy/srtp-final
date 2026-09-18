"""Freeze one margin-protection full-MATH arm; no baseline regeneration."""
import hashlib,io,json,tarfile
from pathlib import Path
import torch
from reflection_margin_reference import cpu_checks,reflection_margin_penalty
from reflection_margin_adapter import device_adjust

HERE=Path(__file__).resolve().parent

def main():
    checks=cpu_checks()
    for dtype in (torch.float32,torch.bfloat16):
        g=torch.Generator().manual_seed(42)
        x=torch.randn(32,63,generator=g).to(dtype)
        ids=[0,1,3,15];mask=torch.rand(32,4,generator=g)>.5
        x[0,-1]=-torch.inf
        expected=reflection_margin_penalty(x,ids,mask,True)
        actual=x.clone();actual[:,ids]=device_adjust(torch,x,torch.tensor(ids),mask)
        assert torch.equal(actual,expected)
        checks.append(str(dtype)+' device implementation matches independent checked reference on mixed masks')
    rel='vcm_signal_audit_20260918/margin_run1'
    folder=HERE/rel;folder.mkdir(exist_ok=False)
    old=json.loads((HERE/'harmonic_confidence_20260918/release.json').read_text())
    prior=json.loads((HERE/'harmonic_confidence_20260918/plan.json').read_text())
    plan={k:prior[k] for k in ('rows','model','dataset','seed','max_new_tokens','temperature','top_p','execution','decision','reference_files')}
    plan.update(experiment_kind='margin_v1',run_id='reflection_margin_math500_20260918_run1',
        arms=[dict(name='MARGIN_L27',suppression_table=rel+'/opening.npz')],new_answers=500,
        hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=2400,
        candidate=dict(formula='lambda=clip(ln2-max(0,max_active_reflection-max_other_finite),0,ln2)',
            attribution='Own design, not official VCM or correctness verifier',
            ordering='raw R confidence saved -> margin replaces L27 magnitude -> original temperature/top_p'),
        interpretation='Repeatedly exposed full benchmark exploration; no independent confirmation or synergy claim',
        engineering=dict(rows=8,arms=['L27_REFERENCE','L27_OFF','L27_SHADOW','MARGIN_L27'],max_tokens=512,
            purpose='native correctness only; do not grade or report accuracy',
            gate='off/shadow exact tokens and control history; actual active change and no pre-intervention divergence'))
    def save(p,d):p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save(folder/'plan.json',plan);save(folder/'cpu_checks.json',dict(checks=checks))
    for n in ('opening.npz','historical_compact.json'):
        (folder/n).write_bytes((HERE/'harmonic_confidence_20260918'/n).read_bytes())
    src=list(old['source_sha256'])+['reflection_margin_reference.py','reflection_margin_adapter.py','prepare_reflection_margin.py']
    release={k:old[k] for k in ('table_schema','assets','engineering_rows','runtime_root','runtime_source_sha256','source_hash_mode')}
    release.update(status='native engineering required before full500',plan_relative_path=rel+'/plan.json',
        plan_sha256=sha(folder/'plan.json'),artifact_root=rel,
        artifact_sha256={n:sha(folder/n) for n in ('opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in src})
    save(folder/'release.json',release)
    archive=HERE.parents[3]/'.codex_work/reflection_margin_20260918_run1.tar.gz'
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in src}
    for n in ('release.json','plan.json','opening.npz','historical_compact.json','cpu_checks.json'):
        files[rel+'/'+n]=(folder/n).read_bytes()
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name,data in files.items():
                entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),checks=checks,engineering_rows=len(release['engineering_rows']))))

if __name__=='__main__':main()
