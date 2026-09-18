import copy,hashlib,io,json,tarfile
from prepare_length_vector import HERE,read,save,sha
from test_counterfactual_admission import main as check

def main():
    check();run='counterfactual_fork_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n',encoding='utf8')
    archive=HERE.parents[3]/'.codex_work/counterfactual_capture_20260918_completed.tar.gz'
    assert sha(archive)=='322e0253becb90abab5ec65c6e202b3046f80a83f58b7b89c0324ccef97ebe45'
    parent='counterfactual_capture_20260918_run2'
    with tarfile.open(archive) as t:
        completion=t.extractfile(parent+'/complete.json').read();gate=json.loads(completion)
        manifest=json.load(t.extractfile(parent+'/capture_manifest.json'))
        assert hashlib.sha256(t.extractfile(parent+'/snapshots.pt').read()).hexdigest()==manifest['snapshot_sha256']
        reference=json.load(t.extractfile(parent+'/reference_complete.json'));observer=json.load(t.extractfile(parent+'/observer_complete.json'));assert reference==observer
    assert gate['passed'] and gate['cases']==8
    prev=HERE/parent;rel=read(prev/'release.json');assert sha(prev/'release.json')==gate['release_sha256']
    plan=read(prev/'plan.json');plan.update(run_id=run,phase='fork_replay',arms=['apply_a','apply_b','skip'],max_new_continuation_tokens=128,
        purpose='Restore same saved L27 prefixes; two no-op forks must match exactly; skip changes only target prefill injection.',
        gate='8 per arm; apply_a/apply_b exact output and control equality, exact prefill masks; skip masks identical except final position zero; at least one nonzero target.',
        limitation='No-op fork equivalence is not equality to original uninterrupted parent numerical partition or RNG trajectory. No causal efficacy claim from this short engineering batch.')
    save(out/'plan.json',plan);(out/'opening.npz').write_bytes((prev/'opening.npz').read_bytes())
    names=list(rel['sources'])+['counterfactual_admission.py'];files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    remote='/root/autodl-tmp/results/hybrid_syncthink_cgrs_20260912/'+parent
    rel.update(sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},plan_sha256=sha(out/'plan.json'),
        snapshot=dict(path=remote+'/snapshots.pt',sha256=manifest['snapshot_sha256'],complete_path=remote+'/complete.json',complete_sha256=hashlib.sha256(completion).hexdigest()))
    save(out/'release.json',rel)
    for n in ('plan.json','release.json','opening.npz'):files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package))))

if __name__=='__main__':main()
