"""Prepare two fixed candidates and all conditional stages while the GPU is off."""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import TRAIN, rows, norm

WORK=ROOT/'.codex_work/overnight_research_20260912'
REMOTE='/root/autodl-tmp/results/easysteer/local_prepared_batch_20260912'
ORDER=['sampled_confidence','lexical_direction']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(not out.exists(),'Immutable parent bundle exists')
    config=ROOT/BASE/'configs'
    prior_path=config/'radial_screen100_fp32_20260912/plan.json';prior=read(prior_path)
    excluded=dict(prior['exclusions'],radial_screen=prior['train_indices'],
        radial_reserve=prior['confirmation']['train_indices'],radial_engineering=prior['engineering']['train_indices'])
    forbidden=set().union(*(set(v) for v in excluded.values()))
    train=rows(ROOT/TRAIN);require(len(train)==7500 and sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Train changed')
    tests={name:rows(ROOT/name) for name in prior['test_prompt_source_sha256']}
    for name,digest in prior['test_prompt_source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Test source changed')
    seen={norm(row['problem']) for group in tests.values() for row in group}|{norm(train[i]['problem']) for i in forbidden}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in forbidden and text not in seen:eligible.append(i);seen.add(text)
    selected=random.Random(20260927).sample(eligible,608)
    engineering=selected[:8];splits={}
    for k,name in enumerate(ORDER):
        part=selected[8+300*k:8+300*(k+1)]
        splits[name]=dict(engineering=engineering,screen=part[:100],confirmation=part[100:])
    overlaps={}
    for name,stage in splits.items():
        for label,indices in stage.items():
            prompt_set={norm(train[i]['problem']) for i in indices}
            for old,values in excluded.items():overlaps[f'{name}_{label}_vs_{old}']=len(set(indices)&set(values))
            for old,values in tests.items():overlaps[f'{name}_{label}_vs_{Path(old).parent.name}']=len(prompt_set&{norm(r['problem']) for r in values})
    require(len(set(selected))==608 and not any(overlaps.values()),'Overlapping selection')
    sampled_cpu=WORK/'sampled_runtime_CPU_committed.json';lexical_cpu=WORK/'lexical_direction_CPU/audit.json'
    require(read(sampled_cpu)['status']=='passed_local_numpy_backed_actual_source_checks','Sampled local checks missing')
    require(read(lexical_cpu)['passes_fixed_gate'],'Lexical CPU gate failed')
    backup=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    for filename,key in [('auto_vector.pt','vector_sha256'),('fit.json','fit_sha256')]:
        require(sha(backup/filename)==prior['arms']['original_dynamic'][key],'Original frozen asset changed')
        require(sha(WORK/'lexical_direction_CPU'/filename)==read(lexical_cpu)['candidate_files_sha256'][filename],'Lexical audited asset changed')
    sampled_receipt=read(sampled_cpu)
    for path,digest in sampled_receipt['source_sha256'].items():require(sha(ROOT/path,source=True)==digest,'Sampled runtime changed after CPU check')
    require(sha(ROOT/BASE/'scripts/check_sampled_confidence_cpu.py',source=True)==sampled_receipt['checker_sha256'],'Sampled CPU checker changed')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n',encoding='utf-8',newline='\n')
    replay=out/'probability_replay'
    subprocess.run([sys.executable,str(ROOT/BASE/'scripts/prepare_sampled_probability_replay.py'),
        '--output',str(replay)],check=True,cwd=ROOT)
    replay_plan=read(replay/'plan.json');replay_plan['output']=REMOTE+'/probability_replay'
    save(replay/'plan.json',replay_plan)
    old_input=WORK/'sampled_probability_inputs100_promptfix_20260912.json'
    require(sha(old_input)=='6b012c58a57c34e104460bcb62191b76a359e48373977494a7d3c5aacaec6ab2','Verified prompt inputs changed')
    expected_inputs=read(old_input);expected_inputs['plan_sha256']=sha(replay/'plan.json')
    save(replay/'expected_inputs.json',expected_inputs)
    source_names=set(prior['source_sha256'])|set(replay_plan['source_sha256'])
    source_names.update(BASE+'scripts/'+name for name in [
        'prepare_local_candidate_batch.py','run_local_prepared_batch.py',
        'check_prepared_prompts.py',
        'check_batch_process_cleanup.py',
        'summarize_local_prepared_batch.py','analyze_pair_uncertainty.py',
        'check_sampled_confidence_cpu.py','check_sampled_confidence_native.py',
        'fit_lexical_direction.py','run_vector_batch.py','grade_vector_batch.py'])
    source_names.update(BASE+'scripts/'+name for name in [
        'audit_lexical_direction_support.py','audit_rebalance_fingerprint.py',
        'audit_prepared_question_similarity.py','audit_accuracy_gate_resolution.py',
        'audit_sampled_curve_coupling.py'])
    source_names.add(BASE+'tests/test_mechanism_screen.py')
    source_names.add('sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/model_runner.py')
    source_hashes={name:sha(ROOT/name,source=True) for name in sorted(source_names)}
    manifests={}
    for candidate in ORDER:
        manifests[candidate]={}
        hypothesis_name='sampled_confidence_20260912.json' if candidate=='sampled_confidence' else 'lexical_direction_20260912.json'
        hypothesis=read(config/hypothesis_name)
        for stage,indices in splits[candidate].items():
            folder=out/candidate/stage;folder.mkdir(parents=True);n=len(indices)
            dataset=folder/'questions.jsonl'
            dataset.write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices),encoding='utf-8',newline='\n')
            assets={}
            for name in ['original_dynamic',candidate]:
                src=WORK/'lexical_direction_CPU' if name=='lexical_direction' else backup
                asset=folder/'assets'/name;asset.mkdir(parents=True)
                for filename in ('auto_vector.pt','fit.json'):shutil.copyfile(src/filename,asset/filename)
                assets[name]=dict(directory='assets/'+name,vector_sha256=sha(asset/'auto_vector.pt'),fit_sha256=sha(asset/'fit.json'))
            if candidate=='sampled_confidence':assets[candidate]['sampled_confidence']=True
            receipt=sampled_cpu if candidate=='sampled_confidence' else lexical_cpu
            shutil.copyfile(receipt,folder/'cpu_receipt.json')
            runtime=dict(prior['runtime'],group_timeout_seconds=300 if stage=='engineering' else (900 if stage=='confirmation' else 600))
            if stage=='engineering':runtime['max_tokens']=512
            plan=dict(status='prepared_not_run',stage=stage,local_prepared_candidate=candidate,
                scope=f'{candidate}: fixed {stage} on {n} disjoint MATH training questions',
                count=n,new_answers_planned=2*n,dataset_file='questions.jsonl',dataset_sha256=sha(dataset),train_indices=indices,
                selection_seed=20260927,overlap_checks=overlaps,
                model=prior['model'],model_files_sha256=prior['model_files_sha256'],decoder_output_layer=20,
                dynamic_parameters=prior['dynamic_parameters'],runtime=runtime,arms=assets,run_order=['original_dynamic',candidate],
                source_sha256=source_hashes,cpu_receipt=dict(file='cpu_receipt.json',sha256=sha(folder/'cpu_receipt.json')),
                hypothesis=hypothesis,expected_gpu_minutes=[2,5] if stage=='engineering' else ([8,18] if stage=='confirmation' else [6,12]),
                batch_timeout_seconds=900 if stage=='engineering' else (2400 if stage=='confirmation' else 1500),
                default_server_output=REMOTE+'/'+candidate+'/'+stage,
                assessment='Engineering: implementation only, no grading. Screen/confirmation: mean thinking AND total tokens each fall>=5percent, correct count not lower, caps not higher. All errors and capped outputs included. No tuning or replacement subsets.',
                authorization='Prepared locally while GPUoff. Execution requires user explicitly reopening GPU and authorizing this bounded batch. No inherited overnight GPU authorization.',
                limitations=['Training screening selects candidates; independent reserve confirms unchanged settings. Neither proves narrow accuracy non-inferiority.','Original and candidate are new paired runs; no historical control reuse.','Lexical fit inherits original selected-layer metadata and original curve; inherited selection.r2 describes original selection, not new lexical-label prediction.'])
            if candidate=='sampled_confidence':plan['probability_gate']=dict(path=REMOTE+'/probability_opportunity.json',replay_plan_sha256=sha(replay/'plan.json'))
            if stage!='engineering':plan['engineering_gate']=dict(path=REMOTE+'/'+candidate+'/engineering/run_ledger.json',
                plan_sha256=manifests[candidate]['engineering']['plan_sha256'])
            if stage=='confirmation':plan['screen_gate']=dict(path=REMOTE+'/'+candidate+'/screen/analysis.json',
                plan_sha256=manifests[candidate]['screen']['plan_sha256'])
            save(folder/'plan.json',plan)
            manifests[candidate][stage]=dict(bundle=str(folder.relative_to(out)).replace('\\','/'),
                plan_sha256=sha(folder/'plan.json'),output=plan['default_server_output'],count=n,
                new_answers=2*n,batch_timeout_seconds=plan['batch_timeout_seconds'])
    parent=dict(status='prepared_GPUoff_waiting_for_explicit_reopening',
        local_phase_start_utc='2026-09-12T05:47:48Z',minimum_four_hour_finish_utc='2026-09-12T09:47:48Z',
        candidate_order=ORDER,candidates=manifests,default_server_output=REMOTE,
        probability_replay=dict(bundle='probability_replay',plan_sha256=sha(replay/'plan.json'),
            expected_inputs_sha256=sha(replay/'expected_inputs.json'),questions=100,saved_thinking_tokens=464036,new_answers=0,timeout_seconds=900),
        native_check_timeout_seconds=180,CPU_preparation_timeout_seconds=180,CPU_analysis_timeout_seconds=180,
        total_timeout_seconds=5400,estimated_minutes_screening=[15,35],estimated_additional_minutes_if_both_confirm=[16,36],
        expected_cost='GPU hourly price is not verified. Cost equals actual instance GPU wall hours times the console hourly rate. The batch ceiling is1.5hours; instance boot, synchronization, transfer and user idle time are additional. No power or mode change is automated.',
        maximum_new_engineering_outputs=32,maximum_new_screen_answers=400,maximum_conditional_confirmation_answers=800,
        selection_seed=20260927,eligible_count=len(eligible),unique_selected_training_questions=608,
        engineering_shared_between_candidates_only=True,splits=splits,exclusions=excluded,overlap_checks=overlaps,
        train_sha256=prior['train_sha256'],test_prompt_source_sha256=prior['test_prompt_source_sha256'],
        original_exclusion_plan_sha256=sha(prior_path),source_sha256=source_hashes,
        execution_policy='One probability replay of saved answers; gate failure skips sampled candidate. For each eligible candidate run8x2 short engineering,100x2 fresh screen, author-grade, and only a passing screen proceeds unchanged to200x2 fixed independent confirmation. Any runtime/input/grade failure stops entire batch with partials retained. No retries,7B,fulltest,AIME orSEAL comparison reruns.',
        stop_policy='90-minute whole-batch ceiling. Do not begin a generation stage with<300seconds remaining for engineering/screen or<600seconds for confirmation. Mark unstarted stages as budget-limited, not efficacy failures. Parent bounds and kills only its own child process groups on timeouts.',
        GPU_authorization='NOT YET PROVIDED FOR THIS NEW BATCH. User has closed GPU and requested four-hour local preparation; wait for explicit reopening.',
        provenance='All fits, input splits, candidate rules and judgments fixed before any new model forward. CPU saved-path statistics are not generation benefits.')
    save(out/'plan.json',parent)
    subprocess.run([sys.executable,str(ROOT/BASE/'scripts/check_prepared_prompts.py'),
        '--bundle',str(out),'--output',str(out/'prompt_audit.json')],check=True,cwd=ROOT)
    parent['prompt_audit']=dict(file='prompt_audit.json',sha256=sha(out/'prompt_audit.json'),
        unique_prompts=608,native_reference_count=100,
        requirement='All new prompt IDs must match the native production tokenizer before any model forward.')
    save(out/'plan.json',parent)
    print(dict(status=parent['status'],bundle=str(out),plan_sha256=sha(out/'plan.json'),eligible=len(eligible),maximum_new_answers=1232))


if __name__=='__main__':main()
