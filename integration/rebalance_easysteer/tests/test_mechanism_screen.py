import copy
import json
from pathlib import Path,PurePosixPath
import sys
import unittest
import tempfile
import shutil
import importlib.util
import argparse
import ast
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout
import io
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mechanism_candidates import ROOT,BASE,read,read_vector,sha
from prepare_mechanism_screen import select,norm
from run_mechanism_screen import command,validate_arm,validate_bundle
from grade_mechanism_screen import compare
from run_vector_batch import command as vector_command, validate_bundle as vector_bundle
from grade_vector_batch import comparison as vector_comparison,controlled_comparison


class ScreenTests(unittest.TestCase):
    def test_greedy_engineering_requires_identical_tokens_including_capped_tails(self):
        from run_vector_batch import engineering_pair_check
        plan=dict(greedy_identity_engineering=True,stage='engineering',
            local_prepared_candidate='sampled_confidence',runtime=dict(temperature=0.0))
        original=dict(rebalance_dynamic=dict(records=[dict(token_ids=[1]*512) for _ in range(8)]))
        candidate=copy.deepcopy(original)
        self.assertEqual(engineering_pair_check(plan,original,candidate)['status'],'greedy_token_identity_passed')
        candidate['rebalance_dynamic']['records'][-1]['token_ids'][-1]=2
        with self.assertRaisesRegex(ValueError,'Greedy identity failed'):
            engineering_pair_check(plan,original,candidate)
        self.assertIsNone(engineering_pair_check({},original,candidate))

    def test_local_summary_keeps_failed_and_partial_candidates_and_rejects_changed_files(self):
        from summarize_local_prepared_batch import summarize
        from mechanism_candidates import save
        names=['sampled_confidence','lexical_direction']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';out=root/'outputs';bundle.mkdir();out.mkdir()
            plan=dict(candidate_order=names,candidates={},source_sha256={BASE+'scripts/'+name:
                sha(ROOT/BASE/'scripts'/name,source=True) for name in
                ['summarize_local_prepared_batch.py','analyze_pair_uncertainty.py']})
            for name in names:
                plan['candidates'][name]={}
                for stage in ('engineering','screen','confirmation'):
                    folder=bundle/name/stage;folder.mkdir(parents=True)
                    save(folder/'plan.json',dict(train_indices=[11,12,13]))
                    plan['candidates'][name][stage]=dict(bundle=name+'/'+stage,count=3,new_answers=6,
                        plan_sha256=sha(folder/'plan.json'))
            save(bundle/'plan.json',plan)
            first=out/names[0]/'screen';first.mkdir(parents=True)
            save(first/'run_ledger.json',dict(status='generation_completed_grading_pending',
                plan_sha256=plan['candidates'][names[0]]['screen']['plan_sha256']))
            summary=dict(generation_seconds=1,preemptions=0,dynamic_kv_replay={})
            groups={name:dict(summary=summary,records=[dict(tokens=n,thinking_tokens=n-10,finish_reason='stop')]*3)
                for name,n in [('original_dynamic',100),(names[0],80)]}
            grades={name:dict(correct=n,seconds=1,records=[dict(correct=i<n) for i in range(3)])
                for name,n in [('original_dynamic',2),(names[0],1)]}
            save(first/'analysis.json',dict(status='completed',stage='screen',candidate=names[0],
                plan_sha256=plan['candidates'][names[0]]['screen']['plan_sha256'],
                comparison=vector_comparison(groups,grades,[11,12,13])))
            partial=out/names[1]/'engineering';partial.mkdir(parents=True)
            save(partial/'run_ledger.json',dict(status='incomplete',
                plan_sha256=plan['candidates'][names[1]]['engineering']['plan_sha256']))
            save(out/'ledger.json',dict(status='incomplete',plan_sha256=sha(bundle/'plan.json'),commit='fixture',
                candidates={names[0]:dict(status='stopped_screen_gate_failed_reserve_untouched'),names[1]:dict(status='pending')},
                files_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},
                children=[],elapsed_seconds=1))
            result=summarize(bundle,out)
            self.assertEqual(result['batch_status'],'incomplete')
            self.assertFalse(result['candidates'][names[0]]['stages']['screen']['metrics']['passes_fixed_gate'])
            self.assertEqual(result['candidates'][names[0]]['stages']['confirmation']['status'],'not_started')
            self.assertEqual(result['candidates'][names[1]]['stages']['engineering']['status'],'incomplete')
            self.assertEqual(result['candidates'][names[1]]['status'],'incomplete_runtime_or_input_failure')
            self.assertNotIn('metrics',result['candidates'][names[1]]['stages']['engineering'])
            (first/'analysis.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Downloaded output differs'):summarize(bundle,out)

    def test_exact_five_percent_gate_does_not_round_nearby_token_totals(self):
        summary=dict(generation_seconds=1,preemptions=0,dynamic_kv_replay={})
        for candidate_tokens,expected in [(1899,True),(1900,True),(1901,False)]:
            groups={name:dict(summary=summary,records=[dict(tokens=n,thinking_tokens=n,finish_reason='stop')])
                for name,n in [('original',2000),('candidate',candidate_tokens)]}
            grades={name:dict(correct=1,seconds=1,records=[dict(correct=True)]) for name in groups}
            result=vector_comparison(groups,grades,[0])
            self.assertEqual(result['passes_fixed_gate'],expected)

    def test_new_vector_command_explicitly_uses_fitted_layer(self):
        bundle=ROOT/BASE/'configs/feedback_controlled_screen100_20260912'
        # Historical assets/commands remain testable; they are not executable
        # evidence for the current source revision.
        plan,_=vector_bundle(bundle,check_source=False)
        for name in plan['run_order']:
            cmd=vector_command(plan,bundle,Path('/out'),name)
            self.assertEqual(cmd[cmd.index('--layer')+1],'20')
            self.assertEqual(cmd[cmd.index('--limit')+1],'100')
            self.assertNotIn('confirmation200.jsonl',' '.join(cmd))
            self.assertEqual('--feedback-config' in cmd,name!='original_dynamic')
            self.assertEqual('--feedback-disabled' in cmd,name=='feedback_disabled')

    def test_graph_control_blocks_apparent_gain_against_only_old_graph(self):
        summary=dict(generation_seconds=1,preemptions=0,dynamic_kv_replay={})
        groups={name:dict(summary=summary,records=[dict(tokens=n,thinking_tokens=n-10,finish_reason='stop') for _ in range(100)])
            for name,n in [('original_dynamic',100),('feedback_disabled',70),('latent_feedback_clip',80)]}
        grades={name:dict(correct=100,seconds=1,records=[dict(correct=True) for _ in range(100)]) for name in groups}
        result=controlled_comparison(groups,grades,list(range(100)))
        self.assertTrue(result['comparisons']['original_dynamic']['passes_fixed_gate'])
        self.assertFalse(result['comparisons']['feedback_disabled']['passes_fixed_gate'])
        self.assertFalse(result['passes_fixed_gate'])

    def test_general_pair_metrics_keep_all_200_errors_and_caps(self):
        summary=dict(generation_seconds=1,preemptions=0,dynamic_kv_replay={})
        x=[dict(tokens=100,thinking_tokens=90,finish_reason='stop') for _ in range(200)]
        x[0]=dict(tokens=16000,thinking_tokens=16000,finish_reason='length')
        y=copy.deepcopy(x)
        for r in y[1:]:r.update(tokens=50,thinking_tokens=40)
        grades=dict(correct=199,seconds=1,records=[dict(correct=i!=0) for i in range(200)])
        worse=copy.deepcopy(grades);worse['correct']=198;worse['records'][3]['correct']=False
        result=vector_comparison({'original_dynamic':dict(records=x,summary=summary),'candidate':dict(records=y,summary=summary)},
            {'original_dynamic':grades,'candidate':worse},list(range(200)))
        self.assertEqual(result['groups']['original_dynamic']['mean_total_tokens'],179.5)
        self.assertEqual(result['groups']['candidate']['capped'],1)
        self.assertEqual(result['degraded_indices'],[3])
        self.assertFalse(result['passes_fixed_gate'])

    def test_split_excludes_indices_and_whitespace_duplicates(self):
        train=[dict(problem='Question '+str(i)) for i in range(500)]
        train[4]['problem']='Question\n 0'
        screen,reserve,_=select(train,['Question 1'],{0,2,3})
        self.assertEqual((len(screen),len(reserve)),(100,200))
        self.assertFalse(set(screen)&set(reserve))
        self.assertFalse(set(screen+reserve)&{0,1,2,3,4})
        self.assertEqual(len({norm(train[i]['problem']) for i in screen+reserve}),300)
        self.assertEqual(select(train,['Question 1'],{0,2,3})[0],screen)

    def fixture(self):
        runtime=dict(max_tokens=16000,temperature=.7,async_scheduling=True)
        plan=dict(runtime=runtime,dynamic_parameters={'q25c':.8},model='/model',dataset_sha256='data',
            arms={'original_dynamic':dict(vector_sha256='vec',fit_sha256='fit',directory='assets/original_dynamic')})
        rows=[dict(problem='q'+str(i),answer=str(i),train_index=i+1000) for i in range(100)]
        records=[dict(dataset_index=i,problem=r['problem'],gold=r['answer'],token_ids=[1]*10,tokens=10,
                      thinking_tokens=8,finish_reason='stop') for i,r in enumerate(rows)]
        saved=dict(status='diagnostic_completed',protocol=dict(runtime,offset=0,limit=100,model='/model',
            run_order=['rebalance_dynamic'],easysteer_output_layer=20,dynamic_params={'q25c':.8}),
            provenance=dict(dataset_sha256='data',vector_sha256='vec',calibration_fit_sha256='fit',git_status=''),
            rebalance_dynamic=dict(records=records,summary=dict(generation_seconds=2,grading_errors=0,preemptions=0,dynamic_kv_replay={})))
        return plan,rows,saved

    def test_pair_validation_rejects_missing_mismatched_or_overcap_answers(self):
        plan,rows,saved=self.fixture();validate_arm(saved,plan,rows,'original_dynamic')
        mutations=[lambda s:s['rebalance_dynamic']['records'].pop(),
                   lambda s:s['rebalance_dynamic']['records'][0].update(problem='wrong'),
                   lambda s:s['rebalance_dynamic']['records'][0].update(tokens=16001,token_ids=[1]*16001),
                   lambda s:s['protocol']['dynamic_params'].update(inject_first_step=True),
                   lambda s:s['provenance'].update(vector_sha256='wrong'),
                   lambda s:s['protocol'].update(async_scheduling=False)]
        for mutate in mutations:
            changed=copy.deepcopy(saved);mutate(changed)
            with self.assertRaises(ValueError):validate_arm(changed,plan,rows,'original_dynamic')

    def test_command_never_runs_reserve_or_unsteered_control(self):
        plan,_,_=self.fixture()
        cmd=command(plan,PurePosixPath('/bundle'),PurePosixPath('/out'),'original_dynamic',PurePosixPath('/repo'))
        self.assertIn('/bundle/screen100.jsonl',cmd)
        self.assertNotIn('confirmation200.jsonl',' '.join(cmd))
        self.assertEqual(cmd[cmd.index('--diagnostic-group')+1],'rebalance_dynamic')
        self.assertEqual(cmd[cmd.index('--max-tokens')+1],'16000')
        self.assertIn('--async-scheduling',cmd)

    def test_analysis_keeps_errors_and_caps_and_catches_accuracy_loss(self):
        _,rows,saved=self.fixture();a=saved['rebalance_dynamic'];b=copy.deepcopy(a)
        a['records'][0].update(tokens=16000,thinking_tokens=16000,finish_reason='length')
        b['records'][0].update(tokens=16000,thinking_tokens=16000,finish_reason='length')
        for r in b['records'][1:]:r.update(tokens=5,thinking_tokens=4)
        ga=dict(correct=99,seconds=1,records=[dict(correct=i!=0) for i in range(100)])
        gb=copy.deepcopy(ga);gb['correct']=98;gb['records'][1]['correct']=False
        result=compare(a,b,ga,gb,[r['train_index'] for r in rows])
        self.assertEqual(result['groups']['original_dynamic']['mean_total_tokens'],169.9)
        self.assertEqual(result['groups']['candidate']['capped'],1)
        self.assertEqual(result['degraded_indices'],[1]);self.assertFalse(result['passes_fixed_screen'])
        json.dumps(result,allow_nan=False)

    def test_accuracy_loss_alone_blocks_otherwise_passing_compression(self):
        _,rows,saved=self.fixture();a=saved['rebalance_dynamic'];b=copy.deepcopy(a)
        for r in b['records']:r.update(tokens=5,thinking_tokens=4)
        ga=dict(correct=100,seconds=1,records=[dict(correct=True) for _ in rows])
        gb=copy.deepcopy(ga)
        self.assertTrue(compare(a,b,ga,gb,list(range(100)))['passes_fixed_screen'])
        gb['correct']=99;gb['records'][0]['correct']=False
        self.assertFalse(compare(a,b,ga,gb,list(range(100)))['passes_fixed_screen'])

    def test_prepared_assets_pass_actual_easysteer_payload_schema(self):
        bundle=ROOT/BASE/'configs/mechanism_screen100_20260911'
        # Historical bundle sources are immutable snapshots; this test checks
        # asset interchange, not whether today's source equals that old run.
        plan=read(bundle/'plan.json')
        path=ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/payloads.py'
        spec=importlib.util.spec_from_file_location('cpu_payload_contract',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=bundle/'assets/original_dynamic/auto_vector.pt'
        for arm in plan['arms'].values():
            source=bundle/arm['directory']/'auto_vector.pt'
            self.assertEqual(sha(source),arm['vector_sha256'])
            vector=read_vector(source,original)
            payload=module.DirectionVector({20:vector})
            wire=payload.to_wire()
            np.testing.assert_array_equal(payload.layers[20],vector)
            self.assertEqual(wire['kind'],'direction')

    def test_bundle_tampering_is_rejected(self):
        bundle=ROOT/BASE/'configs/feedback_controlled_screen100_20260912'
        with tempfile.TemporaryDirectory() as directory:
            copy_path=Path(directory)/'bundle';shutil.copytree(bundle,copy_path)
            vector_bundle(copy_path,check_source=False)
            target=copy_path/'assets/latent_feedback_clip/readout.npy'
            raw=bytearray(target.read_bytes());raw[-20]^=1;target.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'readout changed'):vector_bundle(copy_path,check_source=False)

    def test_historical_bundle_cannot_execute_with_current_changed_source(self):
        bundle=ROOT/BASE/'configs/feedback_controlled_screen100_20260912'
        with self.assertRaisesRegex(ValueError,'Source changed'):
            vector_bundle(bundle)

    def test_sampled_cli_uses_one_factor_and_rejects_combined_interventions(self):
        source=ROOT/BASE/'eval/rebalance_dynamic_eval.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        ns=dict(argparse=argparse,Path=Path,DEFAULT_MODEL='unused',DEFAULT_DATASET='unused',
                DEFAULT_VECTOR='unused',DEFAULT_OUTPUT='unused',__doc__='CPU parser test')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        runtime=dict(max_tokens=16000,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,
            gpu_memory_utilization=.9,temperature=.7,top_p=.95,seed=42,async_scheduling=True,chunked_prefill=False,group_timeout_seconds=600)
        for candidate in ('sampled_confidence','lexical_direction'):
            for stage,n in [('engineering',8),('screen',100),('confirmation',200)]:
                plan=dict(model='model',count=n,dataset_file='questions.jsonl',runtime=dict(runtime),
                    arms={'original_dynamic':dict(directory='original'),candidate:dict(directory=candidate)})
                if candidate=='sampled_confidence':plan['arms'][candidate]['sampled_confidence']=True
                if stage=='engineering':plan['runtime']['max_tokens']=512
                for name in plan['arms']:
                    cmd=vector_command(plan,Path('/bundle'),Path('/out'),name)
                    with patch.object(sys,'argv',cmd[2:]):args=ns['parse_args']()
                    self.assertEqual((args.limit,args.layer,args.max_tokens),(n,20,plan['runtime']['max_tokens']))
                    self.assertEqual(args.sampled_confidence,name=='sampled_confidence')
                    self.assertEqual(args.diagnostic_group,'rebalance_dynamic')
                    self.assertFalse(args.chunked_prefill or args.negative_only or args.feedback_config or args.resume_result)
                    self.assertTrue(args.async_scheduling)
        for extra in [['--negative-only'],['--feedback-config','x'],['--radial-restore','on'],
                      ['--paper-fit','x'],['--resume-result','x'],['--baseline-result','x']]:
            with patch.object(sys,'argv',['eval','--sampled-confidence']+extra),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):ns['parse_args']()

    def test_local_candidate_gate_checks_keep_reserve_conditional(self):
        from run_local_prepared_batch import candidate_eligible,confirmation_eligible
        self.assertFalse(candidate_eligible('sampled_confidence',dict(passes_fixed_gate=False)))
        self.assertTrue(candidate_eligible('lexical_direction',dict(passes_fixed_gate=False)))
        for passed in (False,True):
            analysis=dict(status='completed',stage='screen',candidate='sampled_confidence',plan_sha256='fixed',
                          eligible_for_confirmation=passed,comparison=dict(passes_fixed_gate=passed))
            self.assertEqual(confirmation_eligible(analysis,'sampled_confidence','fixed'),passed)
            for key,value in [('candidate','lexical_direction'),('plan_sha256','other'),('stage','confirmation'),('status','incomplete')]:
                with self.assertRaises(ValueError):confirmation_eligible(dict(analysis,**{key:value}),'sampled_confidence','fixed')
            with self.assertRaises(ValueError):confirmation_eligible(dict(analysis,eligible_for_confirmation=not passed),'sampled_confidence','fixed')

    def test_owned_child_cleanup_on_timeout_and_parent_termination(self):
        """Simulate OS delivery; no real signals or processes on Windows."""
        import run_mechanism_screen as runner
        for mode in ('success','timeout','parent_term','term_during_spawn','nonzero_exit'):
            previous=object();handlers={runner.signal.SIGTERM:previous};kills=[];waits=[]
            def install(sig,handler):
                old=handlers[sig];handlers[sig]=handler;return old
            class Child:
                pid=4242
                def wait(self,timeout):
                    waits.append(timeout)
                    if mode=='timeout' and len(waits)<=2:
                        raise runner.subprocess.TimeoutExpired('owned',timeout)
                    if mode=='parent_term' and len(waits)==1:
                        handlers[runner.signal.SIGTERM](runner.signal.SIGTERM,None)
                    return 124 if mode=='nonzero_exit' else 0
            def spawn(*args,**kwargs):
                self.assertTrue(kwargs['start_new_session'])
                if mode=='term_during_spawn':
                    handlers[runner.signal.SIGTERM](runner.signal.SIGTERM,None)
                return Child()
            with tempfile.TemporaryDirectory() as directory,\
                    patch.object(runner.signal,'signal',install),\
                    patch.object(runner.signal,'SIGKILL',9,create=True),\
                    patch.object(runner.subprocess,'Popen',spawn),\
                    patch.object(runner.os,'killpg',lambda pid,sig:kills.append((pid,sig)),create=True):
                if mode in ('success','nonzero_exit'):
                    self.assertEqual(runner.run_child(['owned'],Path(directory)/'child.log',{},20),
                                     124 if mode=='nonzero_exit' else 0)
                elif mode=='timeout':
                    with self.assertRaises(runner.subprocess.TimeoutExpired):
                        runner.run_child(['owned'],Path(directory)/'child.log',{},20)
                else:
                    with self.assertRaises(SystemExit) as caught:
                        runner.run_child(['owned'],Path(directory)/'child.log',{},20)
                    self.assertEqual(caught.exception.code,128+runner.signal.SIGTERM)
            self.assertIs(handlers[runner.signal.SIGTERM],previous)
            self.assertEqual(kills,[] if mode=='success' else
                [(4242,runner.signal.SIGTERM),(4242,9)])

    def test_local_batch_scheduler_skips_failed_gates_and_stops_on_child_failure(self):
        """No real processes: exercise the full coordinator's decisions and ledger."""
        import run_local_prepared_batch as runner
        from mechanism_candidates import save
        for replay_pass,screen_pass,fail_child,budget_after in [
                (False,False,None,None),(True,True,None,None),(True,False,None,None),
                (True,True,'sampled_confidence_engineering',None),
                (True,True,'sampled_confidence_screen_grade',None),
                (True,True,'lexical_direction_confirmation_grade',None),
                (True,True,None,'sampled_confidence_engineering'),
                (True,True,None,'sampled_confidence_screen_grade')]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);bundle=root/'bundle';bundle.mkdir();out=root/'results'
                (bundle/'plan.json').write_text('{}')
                input_data=dict(plan_sha256='replay',prompts=[]);save(root/'inputs.json',input_data)
                plan=dict(default_server_output=str(out),candidate_order=['sampled_confidence','lexical_direction'],
                    total_timeout_seconds=5400,probability_replay=dict(bundle='replay',expected_inputs_sha256=sha(root/'inputs.json'),
                        plan_sha256='replay',timeout_seconds=900),CPU_preparation_timeout_seconds=180,
                    native_check_timeout_seconds=180,CPU_analysis_timeout_seconds=180,candidates={})
                for candidate in plan['candidate_order']:
                    plan['candidates'][candidate]={stage:dict(bundle=candidate+'/'+stage,
                        output=str(out/candidate/stage),plan_sha256=candidate+'-'+stage,batch_timeout_seconds=900)
                        for stage in ('engineering','screen','confirmation')}
                calls=[];elapsed=[0.0]
                def fake_process(cmd,log,env,timeout,grace_seconds):
                    self.assertEqual(grace_seconds,25)
                    label=log.stem;calls.append(label);log.write_text('fake process; no GPU\n')
                    if label==budget_after:
                        elapsed[0]=5101.0 if label.endswith('engineering') else 4801.0
                    if label==fail_child:return 1
                    def value(flag):return cmd[cmd.index(flag)+1]
                    if label=='check_process_cleanup':save(Path(value('--output')),dict(status='nested_timeout_stopped_owned_separate_groups'))
                    elif label=='check_new_prompt_inputs':save(Path(value('--output')),dict(
                        status='all_new_prompts_match_native_production_tokenizer',plan_sha256=sha(bundle/'plan.json')))
                    elif label=='prepare_saved_probability_inputs':save(Path(value('--prepare-inputs')),input_data)
                    elif label=='native_checks':save(Path(value('--output')),dict(status='native_torch_and_request_checks_passed'))
                    elif label=='probability_opportunity':save(Path(value('--output')),dict(status='completed_offline_probability_opportunity_not_generation',plan_sha256='replay',passes_fixed_gate=replay_pass))
                    elif label=='saved_probability_replay':pass
                    else:
                        destination=Path(value('--output'));destination.mkdir(parents=True,exist_ok=True)
                        candidate=destination.parent.name;stage=destination.name
                        if label.endswith('_grade'):
                            save(destination/'analysis.json',dict(status='completed',stage=stage,candidate=candidate,
                                plan_sha256=candidate+'-'+stage,eligible_for_confirmation=screen_pass,
                                comparison=dict(passes_fixed_gate=screen_pass)))
                        else:
                            run=dict(plan_sha256=candidate+'-'+stage,
                                status='engineering_passed_no_efficacy_claim' if stage=='engineering' else 'generation_completed_grading_pending')
                            if stage=='engineering' and candidate=='sampled_confidence':
                                run['engineering_checks']=dict(status='greedy_token_identity_passed')
                            save(destination/'run_ledger.json',run)
                    return 0
                def fake_query(cmd,**kwargs):return 'fixed-code' if cmd[:2]==['git','rev-parse'] else ''
                with patch.object(runner.sys,'platform','linux'),patch.object(runner.subprocess,'check_output',fake_query),\
                        patch.object(runner.time,'monotonic',lambda:elapsed[0]),\
                        patch.object(runner,'run_child',fake_process),redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                    if fail_child:
                        with self.assertRaisesRegex(ValueError,'Child failed'):runner.execute(bundle,out,plan)
                    else:runner.execute(bundle,out,plan)
                ledger=read(out/'ledger.json')
                if fail_child:
                    self.assertEqual(calls[-1],fail_child)
                    self.assertEqual(ledger['status'],'incomplete')
                    self.assertTrue(ledger['gpu_children_stopped'])
                    self.assertNotIn('pending',[item['status'] for item in ledger['candidates'].values()])
                    failed_candidate=next(name for name in plan['candidate_order'] if fail_child.startswith(name))
                    self.assertEqual(ledger['candidates'][failed_candidate]['status'],'incomplete_runtime_or_input_failure')
                else:
                    self.assertEqual(ledger['status'],'bounded_batch_finished')
                    self.assertEqual('sampled_confidence_engineering' in calls,replay_pass)
                    for candidate in plan['candidate_order']:
                        eligible=candidate!='sampled_confidence' or replay_pass
                        self.assertEqual(candidate+'_confirmation' in calls,eligible and screen_pass and budget_after is None)
                    if budget_after:
                        self.assertEqual(ledger['candidates']['sampled_confidence']['status'],'stopped_budget_before_stage')
                        expected='screen' if budget_after.endswith('engineering') else 'confirmation'
                        self.assertEqual(ledger['candidates']['sampled_confidence']['unstarted_stage'],expected)
                        self.assertEqual(ledger['elapsed_seconds'],elapsed[0])
                self.assertFalse(any('engineering_grade' in call for call in calls))


if __name__=='__main__':unittest.main()
