"""Freeze one MATH500 exploratory evaluation of a training-only feedback vector."""
import ast, hashlib, io, json, tarfile
from prepare_length_vector import HERE, read, save, sha


def main():
    run = 'self_feedback_math500_20260918_run1'
    out = HERE/run
    cpu = HERE/'self_feedback_train500_20260918_run1/fitted'
    check = read(cpu/'result.json')
    assert check['support_gate_passed'] and sha(cpu/'norm_vector.pt') == check['vector_sha256']
    out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n')
    prev = HERE/'question_centered_math_20260918_run1'
    plan, rel = read(prev/'plan.json'), read(prev/'release.json')
    plan.update(run_id=run, experiment_kind='self_feedback_vector_v1',
        arms=[dict(name='FEEDBACK_NORM_L27', vector=run+'/norm_vector.pt', suppression_table=run+'/opening.npz')],
        change='Original O/U means restricted to training parents with correct, naturally closed and shorter L27 outcomes. Original step weighting and vector norm preserved; fit, layer and suppression unchanged.',
        stored_direction='Original unsteered over mean minus under mean on eligible training parents',
        interpretation='Previously exposed MATH500: exploratory evaluation, not independent confirmation. No benchmark outcomes used in fitting.',
        engineering=dict(rows=8, arms=['L27_REFERENCE','FEEDBACK_NORM_L27','FEEDBACK_NORM_REPEAT'], max_tokens=512,
            gate='Reference/candidate agree before first injection; candidate repeat tokens and R history exact'),
        advancement='Both mean thinking and total tokens below frozen L27; caps no higher; observed accuracy loss <=2pp vs both L27 and R. Report paired intervals separately. Stop this candidate if any condition fails; no threshold or selection-mask retuning.',
        expected_cost='One1.5B MATH500 candidate arm;7-12min estimated plus engineering/load/grading. No probe forwards.',
        hard_stop_seconds_per_arm=1200, process_hard_stop_seconds=1500)
    plan['decision'] = dict(train_fit_sha256=sha(cpu/'result.json'), advance_to_gsm=plan['advancement'],
                            efficacy_scale='Paired differences in accuracy percentage points and relative mean token counts')
    save(out/'plan.json',plan); save(out/'cpu_checks.json',check)
    for n in ('opening.npz','historical_compact.json'):
        (out/n).write_bytes((prev/n).read_bytes())
    (out/'norm_vector.pt').write_bytes((cpu/'norm_vector.pt').read_bytes())
    names=list(rel['source_sha256'])+['prepare_feedback_eval.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    for n,b in files.items():
        if n.endswith('.py'): ast.parse(b,filename=n)
    rel.update(status='Outcome-conditioned vector; engineering then exploratory MATH500',artifact_root=run,
        plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ('norm_vector.pt','opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    save(out/'release.json',rel)
    for n in ('release.json','plan.json','cpu_checks.json','norm_vector.pt','opening.npz','historical_compact.json'):
        files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package))))


if __name__=='__main__': main()
