"""Lock a single exploratory norm-matched shared-content vector evaluation."""
import ast
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import torch
from prepare_length_vector import HERE, read, save, sha


def main():
    source=HERE/'repetition_span_replay_20260918_run1';result=source/'results'
    gate=read(result/'complete.json');assert gate['old_first_states_exact'] and gate['questions']==33
    geometry=read(result/'geometry.json')['representations']['shared']
    assert geometry['loo_min']>=.95 and geometry['split_cos_quantiles'][2]>=.5 and geometry['source_cos']>=0
    manifest=read(result/'manifest.json');xs=[]
    for r in manifest:
        assert sha(result/r['file'])==r['sha256'];z=np.load(result/r['file'])
        xs.append(z['repeat_shared'].astype(np.float64)-z['earlier_shared'].astype(np.float64))
    d=np.mean(xs,0)
    originalpath=Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908/auto_vector.pt')
    assert sha(originalpath)=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'
    original=torch.load(originalpath,map_location='cpu',weights_only=True)
    vector=torch.from_numpy((d*float(original.double().norm())/np.linalg.norm(d)).astype(np.float32))
    assert abs(float(vector.double().norm()/original.double().norm())-1)<1e-6
    run='repetition_shared_math500_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n')
    torch.save(vector,out/'auto_vector.pt')
    previous=HERE/'self_feedback_math500_20260918_run1';plan=read(previous/'plan.json');release=read(previous/'release.json')
    plan.update(run_id=run,experiment_kind='reviewed_repetition_vector_v1',arms=[dict(name='REPETITION_NORM_L27',vector=run+'/auto_vector.pt',suppression_table=run+'/opening.npz')],
        change='Only steering vector direction changes: question-equal later-minus-earlier mean on identical token block within33 reviewed repeated computations; norm matched to original. Keep frozen layer21, original fit/curve and L27 suppression.',
        stored_direction='later repeated minus earlier computation; empirical negative steering toward earlier representation, not a guaranteed efficient endpoint',
        hypothesis_selection='Shared-content pooling selected after training-only representation diagnostic to control identical token content; not an independent confirmatory hypothesis. First-token candidate remains rejected. No full-span candidate run.',
        engineering=dict(rows=8,max_tokens=512,arms=['L27_REFERENCE','REPETITION_NORM_L27','REPETITION_NORM_REPEAT'],gate='First injection prefix equivalence, repeat tokens and control history exact, complete finite outputs'),
        advancement='Versus frozen L27 both mean thinking and total tokens must decrease, capped count must not increase, observed accuracy loss<=2pp versus L27 and original R. Report20000 paired bootstrap intervals and flips; exposed benchmark, no independent confirmation. Stop candidate on failure; no post-hoc coefficient, labels or sign tuning.',
        expected_cost='One1.5B full MATH500 candidate only,7-12min generation plus about1min engineering; no inference probes or extra forwards.',
        hard_stop_seconds_per_arm=1200,process_hard_stop_seconds=1500,new_answers=500)
    for key in ['parent_selection','selected_training_questions','training_result_sha256']:
        plan.pop(key,None)
    plan['decision']=dict(training_span_manifest_sha256=sha(result/'manifest.json'),advance_to_gsm=plan['advancement'],efficacy_scale='Paired accuracy percentage points and relative mean token counts')
    plan['training_assets']=dict(question_ids=[r['question'] for r in manifest],span_plan_sha256=sha(source/'plan.json'),span_manifest_sha256=sha(result/'manifest.json'),reviewed_pairs_sha256=sha(HERE/'reviewed_repetition_20260918_run1/reviewed_pairs.json'))
    save(out/'plan.json',plan)
    save(out/'cpu_checks.json',dict(parents=33,geometry=geometry,original_norm=float(original.double().norm()),candidate_norm=float(vector.double().norm()),raw_norm=float(np.linalg.norm(d)),vector_sha256=sha(out/'auto_vector.pt'),runtime_controller_unchanged=True))
    for n in ['opening.npz','historical_compact.json']:(out/n).write_bytes((previous/n).read_bytes())
    names=list(release['source_sha256'])+['prepare_repetition_eval.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    release.update(status='Single shared-content vector experiment locked before benchmark',artifact_root=run,plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ['plan.json','cpu_checks.json','auto_vector.pt','opening.npz','historical_compact.json']},source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    save(out/'release.json',release)
    for n in ['release.json','plan.json','cpu_checks.json','auto_vector.pt','opening.npz','historical_compact.json']:files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with tarfile.open(package,'x:gz') as t:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package),plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
