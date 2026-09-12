"""CPU preflight by default; execute only the explicitly approved prepared GPU batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from run_mechanism_screen import PYTHON, GRADER, run_child
from run_vector_batch import validate_bundle as validate_pair
from prepare_mechanism_screen import norm

ORDER=['sampled_confidence','lexical_direction']


def validate(bundle):
    plan=read(bundle/'plan.json')
    require(plan['status']=='prepared_GPUoff_waiting_for_explicit_reopening','Unexpected parent status')
    require(plan['candidate_order']==ORDER and set(plan['candidates'])==set(ORDER),'Candidate scope changed')
    require(plan['total_timeout_seconds']==5400 and plan['maximum_new_engineering_outputs']==32 and
            plan['maximum_new_screen_answers']==400 and plan['maximum_conditional_confirmation_answers']==800,'Budget changed')
    require(plan['unique_selected_training_questions']==608 and not any(plan['overlap_checks'].values()),'Invalid split declaration')
    train_path=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    require(sha(train_path,source=True)==plan['train_sha256'],'Train changed')
    train=[json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    forbidden=set().union(*(set(v) for v in plan['exclusions'].values()))
    excluded_prompts={norm(train[i]['problem']) for i in forbidden}
    for path,digest in plan['test_prompt_source_sha256'].items():
        require(sha(ROOT/path,source=True)==digest,'Test prompt source changed')
        excluded_prompts.update(norm(json.loads(line)['problem']) for line in
            (ROOT/path).read_text(encoding='utf-8').splitlines())
    prompt_info=plan['prompt_audit'];prompt_path=bundle/prompt_info['file']
    require(prompt_path.resolve().is_relative_to(bundle.resolve()) and sha(prompt_path)==prompt_info['sha256'],'Frozen prompt audit changed')
    prompt_audit=read(prompt_path)
    require(prompt_audit['unique_new_prompts']==608 and prompt_audit['native_reference_count']==100,'Wrong local prompt support')
    for path,digest in plan['source_sha256'].items():require(sha(ROOT/path,source=True)==digest,'Source changed: '+path)
    replay=plan['probability_replay'];replay_bundle=bundle/replay['bundle']
    require(sha(replay_bundle/'plan.json')==replay['plan_sha256'],'Replay plan changed')
    require(sha(replay_bundle/'expected_inputs.json')==replay['expected_inputs_sha256'],'Expected prompt inputs changed')
    replay_plan=read(replay_bundle/'plan.json')
    require(read(replay_bundle/'expected_inputs.json')['plan_sha256']==replay['plan_sha256'],'Inputs use another replay plan')
    require(replay['questions']==100 and replay['new_answers']==0 and replay['saved_thinking_tokens']==464036,'Unplanned replay size')
    require(replay_plan['questions']==100 and replay_plan['capped_answers_included']==8,'Wrong saved-answer population')
    all_indices=[];shared_engineering=None
    for candidate in ORDER:
        stages=plan['candidates'][candidate]
        require(list(stages)==['engineering','screen','confirmation'],'Stage order changed')
        for stage,expected in [('engineering',8),('screen',100),('confirmation',200)]:
            entry=stages[stage];folder=(bundle/entry['bundle']).resolve()
            require(folder.is_relative_to(bundle.resolve()),'Child bundle leaves parent')
            require(sha(folder/'plan.json')==entry['plan_sha256'],'Child plan changed')
            child,rows=validate_pair(folder)
            require(all(row==dict(train[row['train_index']],train_index=row['train_index']) for row in rows),
                    'Question or gold differs from fixed training source')
            require(child['local_prepared_candidate']==candidate and child['stage']==stage and
                    child['count']==expected and entry['count']==expected,'Wrong child stage/count')
            require(child['source_sha256']==plan['source_sha256'],'Child source differs')
            require(child['train_indices']==plan['splits'][candidate][stage],'Child split differs')
            require(child['default_server_output']==entry['output']==plan['default_server_output']+'/'+candidate+'/'+stage,'Output routing changed')
            recorded=prompt_audit['records'][candidate+'/'+stage]
            require([r['train_index'] for r in recorded]==child['train_indices'],'Prompt audit has another split')
            if stage=='engineering':
                if shared_engineering is None:shared_engineering=child['train_indices'];all_indices+=shared_engineering
                else:require(shared_engineering==child['train_indices'],'Engineering pair support changed')
            else:all_indices+=child['train_indices']
            if stage!='engineering':
                require(child['engineering_gate']==dict(path=stages['engineering']['output']+'/run_ledger.json',
                    plan_sha256=stages['engineering']['plan_sha256']),'Engineering gate uses another plan or path')
            if stage=='confirmation':
                require(child['screen_gate']==dict(path=stages['screen']['output']+'/analysis.json',
                    plan_sha256=stages['screen']['plan_sha256']),'Confirmation gate uses another screen or path')
            if candidate=='sampled_confidence':
                require(child['probability_gate']==dict(path=plan['default_server_output']+'/probability_opportunity.json',
                    replay_plan_sha256=replay['plan_sha256']),'Wrong probability gate')
    require(len(all_indices)==len(set(all_indices))==608,'Actual splits overlap')
    require(not forbidden&set(all_indices),'Historical question reused')
    selected_prompts={norm(train[i]['problem']) for i in all_indices}
    require(len(selected_prompts)==608 and not selected_prompts&excluded_prompts,'Actual prompt duplicates or historical overlap')
    return plan


def command(script,*args,python=PYTHON):
    return [python,'-u',str(ROOT/BASE/'scripts'/script),*[str(arg) for arg in args]]


def candidate_eligible(candidate,opportunity):
    return candidate!='sampled_confidence' or opportunity['passes_fixed_gate'] is True


def confirmation_eligible(analysis,candidate,screen_hash):
    require(analysis['status']=='completed' and analysis['stage']=='screen' and
            analysis['candidate']==candidate and analysis['plan_sha256']==screen_hash,'Unverified screen analysis')
    require(analysis['eligible_for_confirmation'] is analysis['comparison']['passes_fixed_gate'],'Contradictory screen gate')
    return analysis['eligible_for_confirmation'] is True


def execute(bundle,out,plan):
    require(sys.platform=='linux' and not out.exists(),'Fresh Linux output required')
    require(out.resolve()==Path(plan['default_server_output']).resolve(),'Use the fixed parent output directory')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU task')
    out.mkdir(parents=True)
    started_monotonic=time.monotonic()
    env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env.get('PATH','')
    ledger=dict(status='running',plan_sha256=sha(bundle/'plan.json'),started_unix=time.time(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        children=[],candidates={},GPU_power_or_mode_changes=0)
    save(out/'ledger.json',ledger)
    def remaining():return plan['total_timeout_seconds']-(time.monotonic()-started_monotonic)
    def child(name,cmd,seconds):
        left=remaining();require(left>30,'Whole-batch deadline')
        # Inner runners get ten seconds; outer cleanup must not kill them first.
        item=dict(name=name,command=cmd,started_unix=time.time(),timeout_seconds=min(seconds,left-30))
        ledger['children'].append(item);save(out/'ledger.json',ledger)
        try:
            code=run_child(cmd,out/(name+'.log'),env,item['timeout_seconds'],grace_seconds=25)
            item.update(exit_code=code,process_seconds=time.time()-item['started_unix'])
            require(code==0,'Child failed; retain partial outputs: '+name)
        except BaseException as error:
            item.update(error=repr(error),process_seconds=time.time()-item['started_unix']);raise
        finally:save(out/'ledger.json',ledger)
    try:
        # Existing inference tokenizer validates the locally frozen prompt IDs before any model load.
        replay=bundle/plan['probability_replay']['bundle']
        child('check_process_cleanup',command('check_batch_process_cleanup.py',
            '--output',out/'native_process_cleanup.json'),plan['CPU_preparation_timeout_seconds'])
        require(read(out/'native_process_cleanup.json')['status']=='nested_timeout_stopped_owned_separate_groups',
                'Native process cleanup failed')
        child('check_new_prompt_inputs',command('check_prepared_prompts.py','--bundle',bundle,
            '--native-check','--output',out/'native_prompt_audit.json'),plan['CPU_preparation_timeout_seconds'])
        prompt_check=read(out/'native_prompt_audit.json')
        require(prompt_check['status']=='all_new_prompts_match_native_production_tokenizer' and
                prompt_check['plan_sha256']==sha(bundle/'plan.json'),'Native new-prompt checks failed')
        child('prepare_saved_probability_inputs',command('replay_sampled_probabilities.py',
            '--bundle',replay,'--prepare-inputs',out/'probability_inputs.json'),plan['CPU_preparation_timeout_seconds'])
        require(sha(out/'probability_inputs.json')==plan['probability_replay']['expected_inputs_sha256'],'Server/local prompt IDs differ')
        child('native_checks',command('check_sampled_confidence_native.py','--output',out/'native_checks.json'),plan['native_check_timeout_seconds'])
        require(read(out/'native_checks.json')['status']=='native_torch_and_request_checks_passed','Native checks failed')
        child('saved_probability_replay',command('replay_sampled_probabilities.py','--bundle',replay,
            '--output',out/'probability_replay','--inputs',out/'probability_inputs.json',
            '--expected-input-sha256',plan['probability_replay']['expected_inputs_sha256'],'--execute',python=GRADER),
            plan['probability_replay']['timeout_seconds'])
        child('probability_opportunity',command('audit_sampled_probabilities.py','--bundle',replay,
            '--replay',out/'probability_replay','--output',out/'probability_opportunity.json',python=GRADER),plan['CPU_analysis_timeout_seconds'])
        opportunity=read(out/'probability_opportunity.json')
        require(opportunity['status']=='completed_offline_probability_opportunity_not_generation' and
                opportunity['plan_sha256']==plan['probability_replay']['plan_sha256'],'Unverified opportunity result')
        for candidate in plan['candidate_order']:
            outcome=dict(status='pending',stages={});ledger['candidates'][candidate]=outcome
            if not candidate_eligible(candidate,opportunity):
                outcome['status']='stopped_probability_gate_failed_no_new_answers';save(out/'ledger.json',ledger);continue
            for stage in ('engineering','screen','confirmation'):
                item=plan['candidates'][candidate][stage]
                if stage=='confirmation' and not confirmation_eligible(
                        analysis,candidate,plan['candidates'][candidate]['screen']['plan_sha256']):
                    outcome['status']='stopped_screen_gate_failed_reserve_untouched';break
                if remaining()<(600 if stage=='confirmation' else 300):
                    outcome.update(status='stopped_budget_before_stage',unstarted_stage=stage);break
                pair=bundle/item['bundle'];destination=Path(item['output'])
                label=candidate+'_'+stage
                child(label,command('run_vector_batch.py','--bundle',pair,'--output',destination,'--execute'),item['batch_timeout_seconds'])
                run=read(destination/'run_ledger.json')
                require(run['plan_sha256']==item['plan_sha256'],'Generation used another plan')
                outcome['stages'][stage]=dict(run_ledger_sha256=sha(destination/'run_ledger.json'),status=run['status'])
                if stage=='engineering':require(run['status']=='engineering_passed_no_efficacy_claim','Engineering failed')
                else:
                    child(label+'_grade',command('grade_vector_batch.py','--bundle',pair,
                        '--output',destination,python=GRADER),plan['CPU_analysis_timeout_seconds'])
                    analysis=read(destination/'analysis.json')
                    outcome['stages'][stage].update(analysis_sha256=sha(destination/'analysis.json'),
                        passes_fixed_gate=analysis['comparison']['passes_fixed_gate'])
                    if stage=='confirmation':outcome['status']='confirmed' if analysis['comparison']['passes_fixed_gate'] else 'stopped_confirmation_failed'
                save(out/'ledger.json',ledger)
            save(out/'ledger.json',ledger)
        ledger.update(status='bounded_batch_finished',completed_unix=time.time(),gpu_work_finished=True)
    except BaseException as error:
        for outcome in ledger['candidates'].values():
            if outcome['status']=='pending':
                outcome['status']='incomplete_runtime_or_input_failure'
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time(),gpu_children_stopped=True);raise
    finally:
        ledger['files_sha256']={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*'))
                                if p.is_file() and p!=out/'ledger.json'}
        ledger['elapsed_seconds']=time.monotonic()-started_monotonic
        save(out/'ledger.json',ledger)
    print(json.dumps(dict(status=ledger['status'],GPU_work_finished=True,
        elapsed_seconds=ledger['elapsed_seconds'],candidates=ledger['candidates'],
        next='Tell the user our GPU batch is finished; check for other tasks before advising shutdown. Finish download verification and reporting locally.')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args=parser.parse_args();bundle=args.bundle.resolve();plan=validate(bundle)
    if not args.execute:
        print(json.dumps(dict(status='CPU_parent_and_all_conditional_stages_validated',
            plan_sha256=sha(bundle/'plan.json'),candidate_order=plan['candidate_order'],
            maximum_short_outputs=32,maximum_screen_answers=400,maximum_conditional_confirmation_answers=800,
            hard_wall_minutes=90,model_loads=0,GPU_calls=0,
            authorization='Wait for explicit user reopening and authorization; --execute is never implied by CPU preflight.')));return
    require(args.expected_plan_sha256 and sha(bundle/'plan.json')==args.expected_plan_sha256,'Approved parent plan hash required')
    execute(bundle,Path(plan['default_server_output']),plan)


if __name__=='__main__':main()
