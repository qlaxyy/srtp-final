"""Recover only unfinished identities; never select cases using correctness."""
import hashlib,io,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    archive=HERE.parents[3]/'.codex_work/mti_run4_failed_20260918.tar.gz'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest=='ebd58a9bd1d6b1a31aa3a46b4a3ee58ef741b287d03119e2c448db90983c871b'
    with tarfile.open(archive) as t:
        base='mti_native_20260918_run4/'
        source=t.extractfile(base+'full/MTI_L27/partial.jsonl').read()
        done=[json.loads(x) for x in source.decode().splitlines()]
        assert len(done)==458 and len({r['dataset_index'] for r in done})==458
        engine=t.extractfile(base+'engineering/complete.json').read()
        assert json.loads(engine)['passed']
        assert 'preemption' in json.loads(t.extractfile(base+'full/failure.json').read())['error']
    parent=HERE/'mti_context_20260918/native_run4'
    plan=json.loads((parent/'plan.json').read_text(encoding='utf8'))
    release=json.loads((parent/'release.json').read_text(encoding='utf8'))
    for r in done:
        assert r['problem_sha256']==plan['rows'][r['dataset_index']]['problem_sha256']
        assert len(r['token_ids'])==r['tokens']<=16000
    remaining=sorted(set(range(500))-{r['dataset_index'] for r in done})
    assert len(remaining)==42
    rel='mti_context_20260918/native_recovery_run1';folder=HERE/rel;folder.mkdir(exist_ok=False)
    def save(p,d):p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    plan['run_id']='mti_native_math500_recovery42_20260918_run1'
    plan['execution']['max_num_seqs']=32
    plan['arms'][0]['suppression_table']=rel+'/opening.npz'
    plan.update(recovery_indices=remaining,new_answers=42,
        recovery_provenance=dict(source_archive_sha256=digest,partial_sha256=hashlib.sha256(source).hexdigest(),
            completed_indices=sorted(r['dataset_index'] for r in done),
            selection='Exactly unfinished identities; no labels inspected',
            schedule_change='Original458 use max_num_seqs256; remaining42 use32. Same model,seed42,method,16000 cap. Combined coverage is a mixed-schedule recovery, not an uninterrupted uniform-batch run.',
            timing='Aborted segment has process wall time only and no complete auxiliary telemetry; never impute total pure generation or auxiliary counts.'))
    save(folder/'plan.json',plan)
    for name in ('opening.npz','historical_compact.json'):(folder/name).write_bytes((parent/name).read_bytes())
    release.update(plan_relative_path=rel+'/plan.json',plan_sha256=sha(folder/'plan.json'),artifact_root=rel,
        artifact_sha256={n:sha(folder/n) for n in ('opening.npz','historical_compact.json','plan.json')},
        engineering_parent_release_sha256=sha(parent/'release.json'),
        engineering_parent_relative_path='mti_context_20260918/native_run4/release.json',
        engineering_complete_sha256=hashlib.sha256(engine).hexdigest(),
        status='42 unfinished questions only; recovery schedule explicitly changed')
    release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in release['source_sha256']}
    save(folder/'release.json',release)
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in release['source_sha256']}
    for n in ('release.json','plan.json','opening.npz','historical_compact.json'):files[rel+'/'+n]=(folder/n).read_bytes()
    files['mti_context_20260918/native_run4/release.json']=(parent/'release.json').read_bytes()
    target=HERE.parents[3]/'.codex_work/mti_native_recovery42_20260918.tar.gz'
    with target.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as t:
            for name,data in files.items():
                info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    print(json.dumps(dict(remaining=remaining,archive_sha256=sha(target))))


if __name__=='__main__':main()
