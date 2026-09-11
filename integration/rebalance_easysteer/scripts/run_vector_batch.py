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


def validate_bundle(bundle):
    plan=read(bundle/'plan.json');n=plan['count']
    require(plan['status']=='prepared_not_run' and n in (100,200),'Unplanned size or status')
    require(plan['decoder_output_layer']==20 and plan['runtime']['max_tokens']==16000,'Wrong layer or cap')
    require(plan['run_order'][0]=='original_dynamic' and len(plan['run_order'])==2,'Only fixed paired vector batches')
    require(set(plan['run_order'])==set(plan['arms']) and plan['new_answers_planned']==2*n,'Wrong scope')
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
    def vector(*args,**kwargs):
        require(kwargs['layers']==[20] and kwargs['algorithm']==expected_algorithm,'Wrong parsed layer/algorithm')
        for k,v in plan['dynamic_parameters'].items():require(kwargs['params'][k]==v,'Wrong parsed controller: '+k)
        seen.append(kwargs['layers']);return actual_vector(*args,**kwargs)
    def stop(**kwargs):raise BeforeModel()
    evaluator.VectorSpec=vector;evaluator.LLM=stop
    try:
        for name in plan['run_order']:
            expected_algorithm='rebalance_feedback' if plan['arms'][name].get('feedback_config') else 'rebalance'
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
        require(record['tokens']==len(record['token_ids'])<=16000,'Invalid cap/count')
        require(0<=record['thinking_tokens']<=record['tokens'],'Invalid thinking count')
        require(record['finish_reason'] in ('stop','length'),'Unfinished answer')
    require(saved['rebalance_dynamic']['summary']['generation_seconds']>0,'Missing timing')
    return saved['rebalance_dynamic']


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
                require(saved['protocol']['dynamic_params']==first['protocol']['dynamic_params'],'Boundary/controller changed')
            else:first=saved
            ledger['arms'][name]['raw_sha256']=sha(out/(name+'.json'));save(out/'run_ledger.json',ledger)
        ledger.update(status='generation_completed_grading_pending',completed_unix=time.time(),gpu_work_finished=True)
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'run_ledger.json',ledger)


if __name__=='__main__':main()
