"""Package one fixed-norm question-centered vector experiment; no remote writes."""
import ast,hashlib,io,json,tarfile
from prepare_length_vector import HERE,read,save,sha

def main():
    run='question_centered_math_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n',encoding='utf8')
    prev=HERE/'strict_and_vector_20260918_run1';cpu=HERE/'question_centered_vector_20260918_cpu'
    assert sha(cpu/'norm_vector.pt')=='241fb0871cb5134d84da7722d310d6a833de8beb22f7a95262794b1df9747583'
    check=read(cpu/'result.json');assert check['cpu_screen_pass']
    plan=read(prev/'plan.json');rel=read(prev/'release.json')
    plan.update(run_id=run,experiment_kind='question_centered_vector_v1',
        arms=[dict(name='QCENTER_NORM_L27',vector=run+'/norm_vector.pt',suppression_table=run+'/opening.npz')],
        change='Original OR labels; center each question on union of original O/U rows before class means. Match original norm. Original fit/layer/controller/L27 unchanged.',
        stored_direction='Residual over mean minus residual under mean; no sign reversal',
        interpretation='Exploratory exposed benchmark. CPU fold alignment is not compression evidence. No retuning.',
        engineering=dict(rows=8,arms=['L27_REFERENCE','QCENTER_NORM_L27','QCENTER_NORM_REPEAT'],max_tokens=512,gate='No divergence before first injection; repeat tokens and R history exact'),
        advancement='Both thinking and total tokens below L27; caps no higher; accuracy loss<=2pp against L27 AND R. No GSM if failed; CI separately reported.',
        expected_cost='One full1.5B MATH500 arm:7-12 min generation estimate plus engineering/load/grading; no extra probe forwards.')
    plan['decision']['advance_to_gsm']=plan['advancement']
    save(out/'plan.json',plan);save(out/'cpu_checks.json',check)
    for n in ('opening.npz','historical_compact.json'):(out/n).write_bytes((prev/n).read_bytes())
    (out/'norm_vector.pt').write_bytes((cpu/'norm_vector.pt').read_bytes())
    names=list(rel['source_sha256'])+['prepare_question_centered_run.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    rel.update(status='Question-centered fixed-norm vector; engineering then MATH500',artifact_root=run,
        plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ('norm_vector.pt','opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    save(out/'release.json',rel)
    for n in ('release.json','plan.json','cpu_checks.json','norm_vector.pt','opening.npz','historical_compact.json'):files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package),plan_sha256=sha(out/'plan.json'))))

if __name__=='__main__':main()
