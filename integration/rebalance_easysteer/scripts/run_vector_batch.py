"""Run one immutable 1.5B vector-only batch, with actual-parser preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mechanism_candidates import ROOT, BASE, require, read, save, sha, read_vector
from run_mechanism_screen import PYTHON, runtime_check, run_child


def validate_bundle(bundle,check_source=True):
    plan=read(bundle/'plan.json');n=plan['count']
    local_candidate=plan.get('local_prepared_candidate')
    engineering=bool(local_candidate and plan['stage']=='engineering')
    require(plan['status']=='prepared_not_run' and n in ((8,) if engineering else (100,200)),'Unplanned size or status')
    require(plan['decoder_output_layer']==20 and plan['runtime']['max_tokens']==(512 if engineering else 16000),'Wrong layer or cap')
    arms=3 if plan.get('graph_control_comparison') else 2
    require(plan['run_order'][0]=='original_dynamic' and len(plan['run_order'])==arms,'Only fixed paired or explicit graph-control batches')
    require(set(plan['run_order'])==set(plan['arms']) and plan['new_answers_planned']==arms*n,'Wrong scope')
    if local_candidate:
        require(local_candidate in ('sampled_confidence','lexical_direction'),'Unknown local candidate')
        require(plan['run_order']==['original_dynamic',local_candidate] and arms==2,'Unexpected local pair')
        require(plan['stage'] in ('engineering','screen','confirmation'),'Unknown local stage')
        require(not any(a.get('negative_only') or a.get('feedback_config') or a.get('radial_restore')
                        for a in plan['arms'].values()),'Combined candidate')
        require(not plan['arms']['original_dynamic'].get('sampled_confidence'),'Changed control')
        require(plan['arms'][local_candidate].get('sampled_confidence',False) is (local_candidate=='sampled_confidence'),'Wrong confidence mode')
        if local_candidate=='sampled_confidence':
            for key in ('vector_sha256','fit_sha256'):
                require(plan['arms']['original_dynamic'][key]==plan['arms'][local_candidate][key],'Sampled-confidence assets differ')
        if engineering and local_candidate=='sampled_confidence':
            require(plan.get('greedy_identity_engineering') is True and plan['runtime']['temperature']==0.0,
                    'Sampled engineering must verify greedy identity')
        elif local_candidate:
            require(plan['runtime']['temperature']==0.7 and not plan.get('greedy_identity_engineering'),
                    'Unplanned generation temperature')
        receipt=read(bundle/plan['cpu_receipt']['file'])
        require(sha(bundle/plan['cpu_receipt']['file'])==plan['cpu_receipt']['sha256'],'CPU receipt changed')
        require(receipt['status']==('passed_local_numpy_backed_actual_source_checks' if local_candidate=='sampled_confidence'
                                   else 'completed_CPU_lexical_direction_fit_not_generation'),'Wrong CPU evidence')
        if local_candidate=='lexical_direction':require(receipt['passes_fixed_gate'],'Lexical CPU gate failed')
    else:
        require(not any(a.get('sampled_confidence') for a in plan['arms'].values()),'Unplanned confidence change')
    if plan.get('positive_branch_ablation'):
        require(plan['run_order']==['original_dynamic','negative_only_dynamic'],'Unexpected positive-branch ablation')
        require(not plan['arms']['original_dynamic'].get('negative_only') and plan['arms']['negative_only_dynamic'].get('negative_only') is True,'Wrong ablation flags')
        require(not any(arm.get('feedback_config') for arm in plan['arms'].values()),'Unplanned combined ablation')
        require(sha(bundle/'cpu_tests.log',source=True)==plan['cpu_unit_log_sha256'],'CPU unit receipt changed')
        require(sha(bundle/'cpu_opportunities.json')==plan['cpu_opportunity_sha256'],'CPU opportunity result changed')
        require(read(bundle/'cpu_opportunities.json')['passes_opportunity_gate'],'No positive-branch opportunities')
        for key in ['vector_sha256','fit_sha256']:
            require(plan['arms']['original_dynamic'][key]==plan['arms']['negative_only_dynamic'][key],'Ablation assets differ')
    else:
        require(not any(arm.get('negative_only') for arm in plan['arms'].values()),'Unplanned positive-branch change')
    if plan.get('radial_restoration'):
        require(arms==3 and plan['run_order']==['original_dynamic','radial_disabled','radial_restore'], 'Unexpected radial design')
        require(plan['arms']['radial_disabled'].get('radial_restore')=='off' and plan['arms']['radial_restore'].get('radial_restore')=='on', 'Wrong radial controls')
        require(not plan['arms']['original_dynamic'].get('radial_restore'), 'Original must use additive family')
        require(not any(arm.get('feedback_config') or arm.get('negative_only') for arm in plan['arms'].values()), 'Combined intervention')
        require(read(bundle/'cpu_summary.json')['passes_fixed_cpu_gate'], 'Radial CPU gate failed')
        require(sha(bundle/'cpu_summary.json')==plan['cpu_summary_sha256'], 'CPU receipt changed')
        require(sha(bundle/'cpu_audit.json')==plan['cpu_audit_sha256'], 'CPU independent audit changed')
        require(sha(bundle/'cpu_tests.log',source=True)==plan['cpu_unit_log_sha256'], 'CPU unit receipt changed')
        require(len({arm[key] for arm in plan['arms'].values() for key in ['vector_sha256']})==1, 'Radial vectors differ')
        require(len({arm['fit_sha256'] for arm in plan['arms'].values()})==1, 'Radial fits differ')
    elif arms==3:
        require(plan['run_order']==['original_dynamic','feedback_disabled','latent_feedback_clip'],'Unexpected graph-control design')
        off,on=plan['arms']['feedback_disabled'],plan['arms']['latent_feedback_clip']
        require(off.get('feedback_disabled') is True and not on.get('feedback_disabled'),'Missing matched off control')
        require({k:v for k,v in off.items() if k!='feedback_disabled'}==on,'Off/on assets differ')
    dataset=bundle/plan['dataset_file']
    require(sha(dataset)==plan['dataset_sha256'],'Dataset changed')
    rows=[json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
    require(len(rows)==n and [r['train_index'] for r in rows]==plan['train_indices'],'Dataset order changed')
    require(not any(plan['overlap_checks'].values()),'Input overlap')
    if plan.get('confirmation'):
        require(sha(bundle/plan['confirmation']['dataset_file'])==plan['confirmation']['dataset_sha256'],'Reserve changed')
    original=bundle/plan['arms']['original_dynamic']['directory']/'auto_vector.pt'
    for name,arm in plan['arms'].items():
        path=bundle/arm['directory'];fit=read(path/'fit.json')
        require(sha(path/'auto_vector.pt')==arm['vector_sha256']==fit['vector_sha256'],'Vector mismatch')
        require(sha(path/'fit.json')==arm['fit_sha256'],'Fit mismatch')
        require(fit['parameters']==plan['dynamic_parameters'],'Controller differs')
        require(fit['model']==plan['model'] and fit['decoder_output_layer']==20 and fit['hidden_state_index']==21,'Fit model/layer mismatch')
        read_vector(path/'auto_vector.pt',original)
        if arm.get('feedback_config'):
            config=bundle/arm['feedback_config'];feedback=read(config)
            require(sha(config)==arm['feedback_config_sha256'],'Feedback metadata changed')
            require(sha(config.parent/feedback['readout_file'])==feedback['readout_sha256'],'Feedback readout changed')
            require(feedback['vector_sha256']==arm['vector_sha256'] and feedback['decoder_output_layer']==20,'Feedback vector/layer differs')
    if check_source:
        for name,digest in plan['source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Source changed: '+name)
    return plan,rows


def command(plan,bundle,out,name):
    folder=bundle/plan['arms'][name]['directory']
    cmd=[PYTHON,'-u',str(ROOT/BASE/'eval/rebalance_dynamic_eval.py'),'--model',plan['model'],
        '--dataset',str(bundle/plan['dataset_file']),'--limit',str(plan['count']),'--layer','20',
        '--vector',str(folder/'auto_vector.pt'),'--calibration-fit',str(folder/'fit.json'),
        '--diagnostic-group','rebalance_dynamic','--output',str(out/(name+'.json'))]
    if plan['arms'][name].get('feedback_config'):
        cmd.extend(['--feedback-config',str(bundle/plan['arms'][name]['feedback_config'])])
        if plan['arms'][name].get('feedback_disabled'):
            cmd.append('--feedback-disabled')
    if 'negative_only' in plan['arms'][name]:
        cmd.append('--negative-only' if plan['arms'][name]['negative_only'] else '--no-negative-only')
    if plan['arms'][name].get('sampled_confidence'):
        cmd.append('--sampled-confidence')
    if plan['arms'][name].get('radial_restore'):
        cmd.extend(['--radial-restore',plan['arms'][name]['radial_restore']])
    for key,value in plan['runtime'].items():
        if isinstance(value,bool):
            if value:cmd.append('--'+key.replace('_','-'))
        else:cmd.extend(['--'+key.replace('_','-'),str(value)])
    return cmd


def actual_parser_check(plan,bundle,out):
    sys.path.insert(0,str(ROOT/BASE/'eval'))
    import rebalance_dynamic_eval as evaluator
    class BeforeModel(Exception): pass
    actual_vector,actual_llm,actual_argv=evaluator.VectorSpec,evaluator.LLM,sys.argv
    seen=[]
    expected_algorithm='rebalance'
    expected_negative_only=False
    expected_sampled_confidence=False
    def vector(*args,**kwargs):
        require(kwargs['layers']==[20] and kwargs['algorithm']==expected_algorithm,'Wrong parsed layer/algorithm')
        require(kwargs['params'].get('negative_only',False) is expected_negative_only,'Wrong parsed positive-branch flag')
        require(kwargs['params'].get('sampled_confidence',False) is expected_sampled_confidence,'Wrong parsed sampled-confidence flag')
        for k,v in plan['dynamic_parameters'].items():require(kwargs['params'][k]==v,'Wrong parsed controller: '+k)
        seen.append(kwargs['layers']);return actual_vector(*args,**kwargs)
    def stop(**kwargs):raise BeforeModel()
    evaluator.VectorSpec=vector;evaluator.LLM=stop
    try:
        for name in plan['run_order']:
            expected_algorithm='rebalance_feedback' if plan['arms'][name].get('feedback_config') else 'rebalance'
            if plan['arms'][name].get('radial_restore'):
                expected_algorithm='rebalance_radial' if plan['arms'][name]['radial_restore']=='on' else 'rebalance_radial_disabled'
            expected_negative_only=plan['arms'][name].get('negative_only',False)
            expected_sampled_confidence=plan['arms'][name].get('sampled_confidence',False)
            sys.argv=[str(ROOT/BASE/'eval/rebalance_dynamic_eval.py')]+command(plan,bundle,out/'preflight_no_generation',name)[3:]
            old=len(seen)
            try:evaluator.main()
            except BeforeModel:pass
            else:raise ValueError('Expected pre-model stop')
            require(len(seen)==old+1,'Missing actual steering spec')
    finally:evaluator.VectorSpec=actual_vector;evaluator.LLM=actual_llm;sys.argv=actual_argv
    return dict(status='actual_parser_and_spec_passed_no_model_loaded',layers=seen)


def validate_arm(saved,plan,rows,name):
    require(saved.get('status')=='diagnostic_completed' and 'baseline' not in saved,'Incomplete/unplanned generation')
    p=saved['protocol'];arm=plan['arms'][name]
    require(p.get('radial_restore')==arm.get('radial_restore'), 'Wrong radial mode')
    if plan.get('radial_restoration'):
        expected={'original_dynamic':'rebalance','radial_disabled':'rebalance_radial_disabled','radial_restore':'rebalance_radial'}[name]
        require(p.get('steering_algorithm')==expected, 'Wrong radial algorithm')
    require(p.get('negative_only',False) is arm.get('negative_only',False),'Wrong positive-branch option')
    require(p['dynamic_params'].get('negative_only',False) is arm.get('negative_only',False),'Wrong request positive-branch option')
    require(p.get('sampled_confidence',False) is arm.get('sampled_confidence',False),'Wrong sampled-confidence protocol')
    require(p['dynamic_params'].get('sampled_confidence',False) is arm.get('sampled_confidence',False),'Wrong sampled-confidence request')
    if arm.get('negative_only'):
        counts=saved['rebalance_dynamic']['summary']['positive_suppression']
        require(counts['boundary_updates_cancelled']>=0 and counts['positive_coefficient_sum']>=0,'Invalid positive-update counters')
    for key,value in plan['runtime'].items():require(p[key]==value,'Runtime differs: '+key)
    require(p['model']==plan['model'] and p['offset']==0 and p['limit']==len(rows)==plan['count'],'Wrong slice')
    require(p['run_order']==['rebalance_dynamic'] and p['easysteer_output_layer']==20,'Wrong parsed layer/group')
    require(not p.get('profiling_enabled') and not p.get('repeat_gate'),'Unplanned observer')
    if arm.get('feedback_config'):
        require(p.get('steering_algorithm')=='rebalance_feedback' and
                p.get('feedback_enabled') is (not arm.get('feedback_disabled',False)),
                'Feedback enable flag differs')
        require(saved['provenance']['feedback_config_sha256']==arm['feedback_config_sha256'],'Wrong feedback config')
        metrics=saved['rebalance_dynamic']['summary']['feedback_applications']
        require(0<=metrics['negative_cancelled']<=metrics['coefficient_changed']<=metrics['negative_executions'],'Invalid feedback counters')
    else:
        require('feedback' not in saved,'Unexpected feedback')
    for k,v in plan['dynamic_parameters'].items():require(p['dynamic_params'][k]==v,'Controller differs')
    require(not p['dynamic_params'].get('inject_first_step',False),'Unplanned first-step injection')
    for k,v in [('dataset_sha256',plan['dataset_sha256']),('vector_sha256',arm['vector_sha256']),('calibration_fit_sha256',arm['fit_sha256'])]:
        require(saved['provenance'][k]==v,'Input differs: '+k)
    require(not saved['provenance']['git_status'].strip(),'Dirty generation source')
    records=saved['rebalance_dynamic']['records'];require(len(records)==len(rows),'Missing answers')
    for i,(record,row) in enumerate(zip(records,rows,strict=True)):
        require(record['dataset_index']==i and record['problem']==row['problem'] and record['gold']==row['answer'],'Pairing mismatch')
        require(record['tokens']==len(record['token_ids'])<=plan['runtime']['max_tokens'],'Invalid cap/count')
        require(record['finish_reason']!='length' or record['tokens']==plan['runtime']['max_tokens'], 'Inconsistent cap reason')
        require(0<=record['thinking_tokens']<=record['tokens'],'Invalid thinking count')
        require(record['finish_reason'] in ('stop','length'),'Unfinished answer')
    require(saved['rebalance_dynamic']['summary']['generation_seconds']>0,'Missing timing')
    return saved['rebalance_dynamic']


def engineering_pair_check(plan,first,second):
    if not plan.get('greedy_identity_engineering'):
        return None
    require(plan['stage']=='engineering' and plan['local_prepared_candidate']=='sampled_confidence'
            and plan['runtime']['temperature']==0.0,'Unplanned greedy comparison')
    original=first['rebalance_dynamic']['records'];candidate=second['rebalance_dynamic']['records']
    require(len(original)==len(candidate)==8,'Greedy check requires exactly8pairs')
    require(all(x['token_ids']==y['token_ids'] for x,y in zip(original,candidate,strict=True)),
            'Greedy identity failed; retain both arms and stop before stochastic efficacy testing')
    return dict(status='greedy_token_identity_passed',pairs=8,temperature=0.0,
                accuracy_or_compression_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--runtime-check',action='store_true')
    a=p.parse_args();bundle=a.bundle.resolve();out=a.output.resolve();plan,rows=validate_bundle(bundle)
    if a.runtime_check:
        receipt=runtime_check(bundle,plan,rows);receipt['parser']=actual_parser_check(plan,bundle,out)
        print(json.dumps(receipt));return
    commands={name:command(plan,bundle,out,name) for name in plan['run_order']}
    if not a.execute:
        print(json.dumps(dict(status='cpu_bundle_preflight_passed',count=plan['count'],new_answers=plan['new_answers_planned'],commands=commands)));return
    require(sys.platform=='linux' and not out.exists(),'Fresh Linux server output required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU process')
    if plan.get('local_prepared_candidate'):
        if plan['local_prepared_candidate']=='sampled_confidence':
            gate=read(Path(plan['probability_gate']['path']))
            require(gate['status']=='completed_offline_probability_opportunity_not_generation' and
                    gate['plan_sha256']==plan['probability_gate']['replay_plan_sha256'] and
                    gate['passes_fixed_gate'] and gate['questions']==100 and gate['capped_answers_included']==8,
                    'Saved-probability opportunity gate incomplete/failed')
        if plan['stage']!='engineering':
            receipt=read(Path(plan['engineering_gate']['path']))
            require(receipt['status']=='engineering_passed_no_efficacy_claim' and
                    receipt['plan_sha256']==plan['engineering_gate']['plan_sha256'],'Local-candidate engineering incomplete')
            if plan['local_prepared_candidate']=='sampled_confidence':
                require(receipt.get('engineering_checks',{}).get('status')=='greedy_token_identity_passed',
                        'Greedy identity was not verified')
        if plan['stage']=='confirmation':
            screen=read(Path(plan['screen_gate']['path']))
            require(screen['status']=='completed' and screen['eligible_for_confirmation'] and
                    screen['candidate']==plan['local_prepared_candidate'] and
                    screen['comparison']['passes_fixed_gate'] and
                    screen.get('plan_sha256')==plan['screen_gate']['plan_sha256'],'Screen did not qualify confirmation')
    if plan.get('graph_control_comparison'):
        receipt=read(Path(plan['engineering_completed_receipt']))
        require(receipt['status']=='engineering_passed_no_efficacy_claim' and receipt['plan_sha256']==sha(bundle/'plan.json'),'New controlled engineering check incomplete')
    if plan.get('positive_branch_ablation'):
        receipt=read(Path(plan['engineering_completed_receipt']))
        require(receipt['status']=='engineering_passed_no_efficacy_claim' and receipt['plan_sha256']==sha(bundle/'plan.json'),'Positive-branch engineering incomplete')
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    ledger=dict(status='running',started_unix=time.time(),plan_sha256=sha(bundle/'plan.json'),commands=commands,arms={},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(out/'run_ledger.json',ledger)
    try:
        check=[PYTHON,'-u',str(Path(__file__)),'--bundle',str(bundle),'--output',str(out),'--runtime-check']
        require(run_child(check,out/'runtime_check.log',env,180)==0,'Runtime preflight failed')
        ledger['runtime_check']=json.loads((out/'runtime_check.log').read_text().splitlines()[-1]);save(out/'run_ledger.json',ledger)
        first=None
        for name in plan['run_order']:
            remaining=plan['batch_timeout_seconds']-(time.time()-ledger['started_unix']);require(remaining>0,'Batch deadline')
            started=time.time();code=run_child(commands[name],out/(name+'.log'),env,min(1200,remaining))
            ledger['arms'][name]=dict(exit_code=code,process_seconds=time.time()-started);save(out/'run_ledger.json',ledger)
            require(code==0,'Generation failed; preserve partial output')
            saved=read(out/(name+'.json'));validate_arm(saved,plan,rows,name)
            require(saved['provenance']['commit']==ledger['commit'],'Source changed during generation')
            if first is not None:
                require(saved['environment']==first['environment'],'Environment changed')
                current=dict(saved['protocol']['dynamic_params']);previous=dict(first['protocol']['dynamic_params'])
                if plan.get('positive_branch_ablation'):
                    current.pop('negative_only',None);previous.pop('negative_only',None)
                if plan.get('local_prepared_candidate')=='sampled_confidence':
                    current.pop('sampled_confidence',None);previous.pop('sampled_confidence',None)
                require(current==previous,'Boundary/controller changed')
                check=engineering_pair_check(plan,first,saved)
                if check is not None:ledger['engineering_checks']=check
            else:first=saved
            ledger['arms'][name]['raw_sha256']=sha(out/(name+'.json'));save(out/'run_ledger.json',ledger)
        status=('engineering_passed_no_efficacy_claim' if plan.get('local_prepared_candidate') and
                plan['stage']=='engineering' else 'generation_completed_grading_pending')
        ledger.update(status=status,completed_unix=time.time(),gpu_work_finished=True)
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'run_ledger.json',ledger)


if __name__=='__main__':main()
