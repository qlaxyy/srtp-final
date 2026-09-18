import hashlib,io,json,tarfile
from prepare_length_vector import HERE,read,save,sha
from test_counterfactual_snapshot import main as check

def main():
    check();run='counterfactual_capture_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n',encoding='utf8')
    parent=HERE/'length_sign_ablation_20260918_run1';old=read(parent/'release.json')
    (out/'opening.npz').write_bytes((parent/'opening.npz').read_bytes())
    plan=dict(run_id=run,rows=old['engineering_rows'],model='DeepSeek-R1-Distill-Qwen-1.5B',
        arms=['L27_REFERENCE','L27_SNAPSHOT_OBSERVER'],max_tokens=512,seed=42,temperature=.7,top_p=.95,
        purpose='Eight previously exposed training engineering questions, sixteen short outputs; no calibration regeneration or benchmark efficacy.',
        execution=dict(sync=True,max_num_seqs=16,max_num_batched_tokens=32768,memory=.9,hard_stop_seconds=600),
        estimated_minutes='2-5 including loading, bound 10 minutes; no full test authorized by success alone',
        gate='Eight snapshots; reference/observer exact token and control-record equality; no parent mutation or preemption; preserve failures.',
        new_forward='No diagnostic extra model forwards; normal generation for 16 short outputs only. CPU snapshot overhead included in observer timing.')
    save(out/'plan.json',plan)
    names=['adapter.py','policy.py','label_alignment_adapter.py','replay_adapter.py','replay_label_alignment.py',
        'counterfactual_snapshot.py','counterfactual_observer.py','run_snapshot_engineering.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    identity=dict(model=old['assets']['model_files']['model.safetensors']['sha256'],vector=old['assets']['vector']['sha256'],fit=old['assets']['fit']['sha256'],table=sha(out/'opening.npz'),runtime=old['runtime_source_sha256'])
    release=dict(assets=old['assets'],runtime_root=old['runtime_root'],runtime_source_sha256=old['runtime_source_sha256'],
        identity=identity,sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},table_sha256=sha(out/'opening.npz'),plan_sha256=sha(out/'plan.json'))
    save(out/'release.json',release)
    for n in ('plan.json','release.json','opening.npz'):files[run+'/'+n]=(out/n).read_bytes()
    path=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with path.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tf:
            for n,b in files.items():
                entry=tarfile.TarInfo(n);entry.size=len(b);tf.addfile(entry,io.BytesIO(b))
    print(json.dumps(dict(package=str(path),sha256=sha(path),runtime=release['runtime_root'])))

if __name__=='__main__':main()
