"""Freeze one full-MATH candidate and its short engineering checks."""
import hashlib,json,tarfile,io,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args()
    assert a.attempt>=1
    rel=f'mti_context_20260918/native_run{a.attempt}'
    folder=HERE/rel;folder.mkdir(exist_ok=False)
    old=json.loads((HERE/'harmonic_confidence_20260918/release.json').read_text(encoding='utf8'))
    prior=json.loads((HERE/'harmonic_confidence_20260918/plan.json').read_text(encoding='utf8'))
    plan={k:prior[k] for k in ('rows','model','dataset','seed','max_new_tokens','temperature','top_p','execution','decision','reference_files')}
    plan.update(experiment_kind='mti_v1',run_id=f'mti_native_math500_20260918_run{a.attempt}',
        arms=[dict(name='MTI_L27',suppression_table=rel+'/opening.npz')],
        new_answers=500,hard_stop_seconds_per_arm=3600,process_hard_stop_seconds=4200,
        candidate=dict(entropy_threshold=.5,guidance_scale=1.5,cue='OUTPUT ERROR',
            ordering='raw entropy -> contrast -> frozen L27 -> temperature/top_p; raw pmax retained for native R',
            cue_steering='Cue contains no step-boundary token, so original boundary-only steering applies no fresh vector to cue. Existing steered prefix KV is reused.'),
        interpretation='Repeatedly exposed full benchmark exploration, no independent confirmation or synergy claim',
        expected_full_minutes=[15,60],engineering=dict(rows=8,arms=['L27_REFERENCE','L27_OFF','L27_SHADOW','MTI_L27'],
            max_tokens=64,purpose='native branch integrity only; never report accuracy',
            gate='off/shadow token and R history exact identity, nonzero auxiliary calls, no divergence before first intervention'))
    def save(p,d):p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save(folder/'plan.json',plan)
    for name in ('opening.npz','historical_compact.json'):
        (folder/name).write_bytes((HERE/'harmonic_confidence_20260918'/name).read_bytes())
    src=list(old['source_sha256'])+['mti_reference.py','mti_native_adapter.py']
    release={k:old[k] for k in ('table_schema','assets','engineering_rows','runtime_root','runtime_source_sha256','source_hash_mode')}
    release.update(status='native engineering required before full500',plan_relative_path=rel+'/plan.json',
        plan_sha256=sha(folder/'plan.json'),artifact_root=rel,
        artifact_sha256={n:sha(folder/n) for n in ('opening.npz','historical_compact.json','plan.json')},
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in src})
    save(folder/'release.json',release)
    archive=HERE.parents[3]/f'.codex_work/mti_native_20260918_run{a.attempt}.tar.gz'
    # Runner resolves HERE locally, and uses the existing hash-pinned runtime separately.
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in src}
    for n in ('release.json','plan.json','opening.npz','historical_compact.json'):
        files[rel+'/'+n]=(folder/n).read_bytes()
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name,data in files.items():
                entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),engineering_rows=len(release['engineering_rows'])),ensure_ascii=False))


if __name__=='__main__':main()
