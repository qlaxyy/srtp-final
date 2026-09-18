"""Package each supported endpoint design as one frozen MATH500 arm."""
import ast,hashlib,io,json,tarfile
from prepare_length_vector import HERE,read,save,sha
from outcome_efficient_adapter import cpu_checks

def main():
    source=HERE/'outcome_endpoints_20260918_run1';checks=cpu_checks()
    for arm,mode in [('EFFICIENT','efficient'),('UNDER_REFIT','under')]:
        fitdir=source/'fitted'/arm;report=read(fitdir/'report.json')
        if not report['supported']:continue
        run='outcome_'+mode+'_math500_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
        (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n')
        prev=HERE/'self_feedback_math500_20260918_run1';plan=read(prev/'plan.json');rel=read(prev/'release.json')
        name=arm+'_L27'
        plan.update(run_id=run,experiment_kind='outcome_endpoint_v1',endpoint_mode=mode,
            arms=[dict(name=name,vector=run+'/auto_vector.pt',fit=run+'/fit.json',suppression_table=run+'/opening.npz')],
            change=read(source/'protocol.json')['arms'][arm],stored_direction='Raw long-side mean minus specified short-side mean; no normalization',
            engineering=dict(rows=8,max_tokens=512,arms=['L27_REFERENCE',name,name.replace('_L27','_REPEAT')],gate='Pre-injection prefix and repeat tokens/history; E additionally native vs off/shadow exact'),
            decision=dict(training_protocol_sha256=sha(source/'protocol.json'),fit_report_sha256=sha(fitdir/'report.json'),advance_to_gsm=plan['advancement']),
            expected_cost='One1.5B MATH500 arm7-12min estimate plus engineering; no probe forwards. Two endpoint designs independently stopped if criteria fail.',
            hard_stop_seconds_per_arm=1200,process_hard_stop_seconds=1500)
        save(out/'plan.json',plan);save(out/'cpu_checks.json',dict(controller_checks=checks,fit=report))
        for n in ['opening.npz','historical_compact.json']:(out/n).write_bytes((prev/n).read_bytes())
        for n in ['auto_vector.pt','fit.json']:(out/n).write_bytes((fitdir/n).read_bytes())
        names=list(rel['source_sha256'])+['outcome_efficient_adapter.py','prepare_endpoint_eval.py']
        files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
        for n,b in files.items():
            if n.endswith('.py'):ast.parse(b,filename=n)
        rel.update(status='Endpoint-specific vector and refitted/new control',artifact_root=run,plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
            artifact_sha256={n:sha(out/n) for n in ['auto_vector.pt','fit.json','opening.npz','historical_compact.json','plan.json','cpu_checks.json']},
            source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
        save(out/'release.json',rel)
        for n in ['release.json','plan.json','cpu_checks.json','auto_vector.pt','fit.json','opening.npz','historical_compact.json']:files[run+'/'+n]=(out/n).read_bytes()
        package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
        with package.open('xb') as f:
            with tarfile.open(fileobj=f,mode='w:gz') as t:
                for n,b in files.items():
                    info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
        print(json.dumps(dict(arm=arm,package=str(package),sha256=sha(package))))

if __name__=='__main__':main()
