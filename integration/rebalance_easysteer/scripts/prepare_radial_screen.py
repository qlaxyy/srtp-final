"""Freeze a radial-only 3x100 screen after CPU geometry and behavior checks."""
import argparse
import json
import random
import shutil
from pathlib import Path
from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import rows, norm, TRAIN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--unit-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve()
    require(not out.exists(),'Immutable bundle exists')
    cpu=read(a.cpu/'summary.json');audit=read(a.cpu/'local_audit.json')
    require(cpu['passes_fixed_cpu_gate'] and audit['status']=='independent_law_of_cosines_and_gate_verified','CPU gate/audit failed')
    unit=json.loads(a.unit_log.read_text(encoding='utf-8').splitlines()[-1])
    require(unit['status']=='CPU_tests_passed' and unit['count']==21 and unit['GPU_calls']==0,'CPU behavior checks incomplete')
    config=ROOT/BASE/'configs';hypothesis=config/'radial_restoration_20260912.json'
    require(sha(hypothesis,source=True)==cpu['plan_sha256'],'Fixed hypothesis changed')
    prior=read(config/'trajectory_progress_screen100_20260912/plan.json')
    exclusions=dict(prior['exclusions'],trajectory_progress_screen=prior['train_indices'],
                    trajectory_progress_reserve=prior['confirmation']['train_indices'])
    excluded=set().union(*(set(v) for v in exclusions.values()))
    train=rows(ROOT/TRAIN);require(sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Train source changed')
    tests={name:rows(ROOT/name) for name in prior['test_prompt_source_sha256']}
    for name,digest in prior['test_prompt_source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Test source changed')
    seen={norm(r['problem']) for group in tests.values() for r in group}|{norm(train[i]['problem']) for i in excluded}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:eligible.append(i);seen.add(text)
    chosen=random.Random(20260922).sample(eligible,308)
    engineering,screen,reserve=chosen[:8],chosen[8:108],chosen[108:]
    overlap={}
    for name,indices in [('engineering',engineering),('screen',screen),('confirmation',reserve)]:
        prompts={norm(train[i]['problem']) for i in indices}
        require(len(prompts)==len(indices),'Duplicate prompts')
        for old,values in exclusions.items():overlap[name+'_vs_'+old]=len(set(indices)&set(values))
        for old,group in tests.items():overlap[name+'_vs_'+Path(old).parent.name]=len(prompts&{norm(r['problem']) for r in group})
    require(len(set(chosen))==308 and not any(overlap.values()),'Input overlap')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n*.log text eol=lf\n',encoding='utf-8',newline='\n')
    for file,indices in [('smoke8.jsonl',engineering),('screen100.jsonl',screen),('confirmation200.jsonl',reserve)]:
        (out/file).write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices),encoding='utf-8',newline='\n')
    backup=ROOT/read(hypothesis)['inputs']['backup'];arms={}
    for name in ['original_dynamic','radial_disabled','radial_restore']:
        folder=out/'assets'/name;folder.mkdir(parents=True)
        for file in ['auto_vector.pt','fit.json']:shutil.copyfile(backup/file,folder/file)
        arms[name]=dict(directory='assets/'+name,vector_sha256=sha(folder/'auto_vector.pt'),fit_sha256=sha(folder/'fit.json'))
    arms['radial_disabled']['radial_restore']='off'
    arms['radial_restore']['radial_restore']='on'
    shutil.copyfile(a.cpu/'summary.json',out/'cpu_summary.json')
    save(out/'cpu_audit.json',audit)
    shutil.copyfile(a.unit_log,out/'cpu_tests.log')
    paths=set(prior['source_sha256'])
    paths.update(BASE+name for name in ['eval/baseline_reuse.py','scripts/audit_radial_restoration.py',
        'scripts/prepare_radial_screen.py','scripts/check_radial_graph.py','scripts/run_feedback_engineering.py',
        'scripts/run_vector_batch.py','scripts/grade_vector_batch.py'])
    paths.update('sources/EasySteer/vllm-steer/'+name for name in [
        'vllm/steer_vectors/api.py','vllm/steer_vectors/request.py','vllm/steer_vectors/payloads.py',
        'vllm/steer_vectors/graph_kernels.py','vllm/steer_vectors/algorithms/rebalance.py',
        'vllm/steer_vectors/algorithms/__init__.py','vllm/v1/worker/gpu/steer_vector_utils.py',
        'tests/steer_vectors/test_rebalance.py'])
    plan=dict(status='prepared_not_run',stage='screen',radial_restoration=True,graph_control_comparison=True,
        scope='Fresh100 MATH training radial-only norm restoration; unchanged signed dynamic controller and original vector; original additive and matched radial-disabled controls',
        count=100,new_answers_planned=300,dataset_file='screen100.jsonl',dataset_sha256=sha(out/'screen100.jsonl'),
        selection_seed=20260922,train_indices=screen,eligible_count=len(eligible),exclusions=exclusions,overlap_checks=overlap,
        confirmation=dict(count=200,train_indices=reserve,dataset_file='confirmation200.jsonl',dataset_sha256=sha(out/'confirmation200.jsonl'),status='untouched_conditional_three_arm_reserve'),
        engineering=dict(count=8,train_indices=engineering,dataset_file='smoke8.jsonl',dataset_sha256=sha(out/'smoke8.jsonl'),max_tokens=512,
            order=list(arms),new_short_outputs=24,group_timeout_seconds=300,batch_timeout_seconds=1800,
            purpose='Kernel formula, mutable CUDA graph, complete-state normalization and original/off/on short generation; no efficacy grading.',
            kernel_gate='Compiled vs eager BF16 error <= (abs(hidden)+abs(compiled-hidden)+1)/128; relative complete-state norm error <0.005; captured output exact on enabled/mask mutations; zero mask and cleared row exact identity.',
            short_gate='All24 outputs complete within512, verified model/code/assets/environment/layer/controller; at least1 off/on token stream differs. Original-vs-off numeric differences recorded, not used to choose a control. Both controls mandatory in the screen.',
            stop='Any implementation, source, runtime or deadline failure stops before100-screen. Preserve output; no cap extension or automatic rerun.',expected_gpu_minutes=[3,10]),
        engineering_completed_receipt='/root/autodl-tmp/results/easysteer/radial_engineering8_20260912/ledger.json',
        train_sha256=prior['train_sha256'],test_prompt_source_sha256=prior['test_prompt_source_sha256'],
        model=prior['model'],model_files_sha256=prior['model_files_sha256'],decoder_output_layer=20,
        dynamic_parameters=prior['dynamic_parameters'],runtime=prior['runtime'],arms=arms,run_order=list(arms),
        source_sha256={name:sha(ROOT/name,source=True) for name in sorted(paths)},
        cpu_summary_sha256=sha(out/'cpu_summary.json'),cpu_audit_sha256=sha(out/'cpu_audit.json'),
        cpu_unit_log_sha256=sha(out/'cpu_tests.log',source=True),hypothesis=read(hypothesis),
        assessment='Against EACH control: BOTH thinking and total mean tokens decrease at least5percent, correct count does not fall, caps do not increase. All errors and caps included. Failure stops candidate without strength or norm-factor search; only a passing candidate can use unchanged three-arm confirmation on reserved200.',
        expected_gpu_minutes=[10,20],batch_timeout_seconds=2400,
        stop_conditions=['Mismatch or another GPU task: stop before loading.','Engineering failure: retain outputs and stop.','Incomplete group or deadline: preserve partial answers and stop without auto-retry.','Failure against either control: stop and retain unused200 reserve.'],
        default_server_output='/root/autodl-tmp/results/easysteer/radial_screen100_20260912',
        authorization='User continuous GPU research authorization; fixed1.5B24 short engineering outputs and300 screen answers. No7B or full frozen benchmark rerun.',
        limitations=['Saved-state radial statistics do not predict online compression.','FP32 radial geometry with BF16 displacement is approximate norm preservation, not exact real arithmetic.','A new graph can alter compiler rounding even disabled; both off/on declare the same two radial algorithms.','Screening evidence is not independent accuracy confirmation.'])
    save(out/'plan.json',plan)
    print(json.dumps(dict(screen=100,engineering=8,reserve=200,eligible=len(eligible),plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
