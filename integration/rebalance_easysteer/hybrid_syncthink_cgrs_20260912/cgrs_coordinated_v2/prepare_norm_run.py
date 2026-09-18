"""Exclusive release for complete-state norm correction with old runtime intact."""
import ast,hashlib,io,tarfile,json
from prepare_length_vector import HERE,read,save,sha
from check_norm_graph import check

def main():
    run='norm_preserving_math_20260918_run2';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n',encoding='utf8')
    previous=HERE/'question_centered_math_20260918_run1'
    plan=read(previous/'plan.json');release=read(previous/'release.json')
    plan.update(run_id=run,experiment_kind='norm_preserving_v1',
      arms=[dict(name='NORM_STATE_L27',suppression_table=run+'/opening.npz')],
      change='Only rescale complete decoder output after original additive steering to original norm. Raw vector/layer/fit/L27 fixed.',
      engineering=dict(rows=8,arms=['L27_REFERENCE','NORM_OFF','NORM_STATE_L27','NORM_REPEAT'],max_tokens=512,
        gate='Separate unmodified process reference; disabled token/control equality; active prefix equality and repeat equality; positive active rows and no fallback.'),
      expected_cost='8-15min estimated full generation plus engineering/load/grading. No added model forwards; reduction cost included.',
      hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=2100,
      interpretation='Exposed benchmark exploration only. No efficacy inferred from geometry. No post-failure strength or mix search.')
    save(out/'plan.json',plan)
    checks=check(HERE.parents[3]/'sources/EasySteer/vllm-steer/vllm/steer_vectors/graph_kernels.py')
    save(out/'cpu_checks.json',checks)
    for n in ('opening.npz','historical_compact.json'):(out/n).write_bytes((previous/n).read_bytes())
    names=list(release['source_sha256'])+['prepare_norm_run.py','norm_graph_adapter.py','check_norm_graph.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    release.update(status='Isolated norm graph adapter; unmodified reference then engineering and full MATH500',
        artifact_root=run,plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ('opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    release['norm_reference_release_sha256']=sha(HERE/'norm_preserving_math_20260918_run1/release.json')
    save(out/'release.json',release)
    for n in ('release.json','plan.json','cpu_checks.json','opening.npz','historical_compact.json'):files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package),plan_sha256=sha(out/'plan.json'))))

if __name__=='__main__':main()
