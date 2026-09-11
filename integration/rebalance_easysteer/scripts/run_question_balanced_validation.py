"""One fixed 100-question original/candidate pair; reuse runtime and author grader."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_repeat_validation import (ROOT,BASE,PYTHON,GRADER,require,sha,read,save,
                                   command as old_command,validate_arm as old_validate)


def arm_plan(plan, arm):
    return dict(plan, **plan['calibrations'][arm])


def validate_bundle(bundle):
    p=read(bundle/'plan.json')
    require(p['status']=='prepared_not_run' and p['count']==100,'Not a fixed fresh plan')
    require(p['run_order']==['original_dynamic','question_balanced'],'Wrong comparison')
    require(p['runtime']['max_tokens']==16000,'Wrong cap')
    require(sha(bundle/'validation100.jsonl')==p['dataset_sha256'],'Dataset changed')
    rows=[json.loads(s) for s in (bundle/'validation100.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows)==100 and [r['train_index'] for r in rows]==p['train_indices'],'Order changed')
    require(not any(p['overlap_checks'].values()),'Overlapping questions')
    for name,expected in p['source_sha256'].items():
        require(sha(ROOT/name,source=True)==expected,'Source changed: '+name)
    return p,rows


def combine(original,candidate,plan,rows):
    groups=[old_validate(saved,arm_plan(plan,arm),rows,'off') for saved,arm in
            zip((original,candidate),plan['run_order'],strict=True)]
    require(original['environment']==candidate['environment'],'Environment differs')
    for key in ('commit','git_status'):
        require(original['provenance'][key]==candidate['provenance'][key],'Code differs')
    protocol=dict(original['protocol'],diagnostic_only=False,
        run_order=['baseline','rebalance_dynamic'],
        group_meanings={'baseline':'original dynamic; NOT unsteered',
                        'rebalance_dynamic':'question-balanced class calibration; same runtime'},
        arm_protocols={a:s['protocol'] for a,s in zip(plan['run_order'],(original,candidate),strict=True)})
    return dict(status='completed',scope='question-balanced 1.5B fresh MATH training validation',
        protocol=protocol,baseline=groups[0],rebalance_dynamic=groups[1],environment=original['environment'],
        provenance={a:s['provenance'] for a,s in zip(plan['run_order'],(original,candidate),strict=True)})


def analyze(pair,graded):
    require(set(graded['groups'])=={'baseline','rebalance_dynamic'},'Incomplete grading')
    result=dict(status='completed',scope=pair['scope'],groups={})
    for key,arm in zip(('baseline','rebalance_dynamic'),('original_dynamic','question_balanced'),strict=True):
        records=pair[key]['records']; summary=pair[key]['summary']; grade=graded['groups'][key]
        require(len(records)==len(grade['records'])==100,'Incomplete records')
        result['groups'][arm]=dict(count=100,author_correct=grade['author_correct'],
            author_accuracy_percent=grade['author_correct'],
            mean_total_tokens=sum(r['tokens'] for r in records)/100,
            mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/100,
            capped=sum(r['finish_reason']=='length' or r['tokens']==16000 for r in records),
            generation_seconds=summary['generation_seconds'],preemptions=summary['preemptions'],
            dynamic_kv_replay=summary['dynamic_kv_replay'],grading_errors=summary['grading_errors'])
    a,b=result['groups'].values()
    result.update(improved_indices=graded['improved_indices'],degraded_indices=graded['degraded_indices'],
        accuracy_change_percentage_points=b['author_correct']-a['author_correct'],
        total_token_change_percent=100*(b['mean_total_tokens']/a['mean_total_tokens']-1),
        changed_outputs=sum(x['token_ids']!=y['token_ids'] for x,y in
            zip(pair['baseline']['records'],pair['rebalance_dynamic']['records'],strict=True)),
        decision='No retuning on these 100; a promising result requires separate confirmation, not proof of accuracy preservation.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(); bundle=args.bundle.resolve(); output=args.output.resolve()
    plan,rows=validate_bundle(bundle)
    commands={arm:old_command(arm_plan(plan,arm),bundle/'validation100.jsonl',output/(arm+'.json'),'off') for arm in plan['run_order']}
    if not args.execute:
        print(json.dumps(dict(status='cpu_preflight_passed',commands=commands))); return
    require(sys.platform=='linux','Use existing server runtimes')
    require(not output.exists(),'Fresh output required; preserve partials and never auto-rerun')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit code first')
    for exe in (PYTHON,GRADER): require(Path(exe).is_file(),'Missing existing runtime')
    env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    check="import inspect,json,vllm,torch; from easysteer.vectors import from_pt_direction; print(json.dumps(dict(vllm_file=vllm.__file__,vector_file=inspect.getfile(from_pt_direction),vllm_version=vllm.__version__,torch_version=torch.__version__)))"
    receipt=json.loads(subprocess.check_output([PYTHON,'-c',check],cwd=ROOT,env=env,text=True).splitlines()[-1])
    for key,rel in (('vllm_file','sources/EasySteer/vllm-steer/vllm/__init__.py'),('vector_file','sources/EasySteer/easysteer/vectors.py')):
        require(Path(receipt[key]).resolve()==(ROOT/rel).resolve(),'Wrong editable import path')
    for name,expected in plan['model_files_sha256'].items():
        require(sha(Path(plan['model'])/name)==expected,'Model changed: '+name)
    for arm in plan['run_order']:
        ap=arm_plan(plan,arm)
        for name,key in (('auto_vector.pt','vector_sha256'),('fit.json','fit_sha256')):
            require(sha(Path(ap['assets'])/name)==ap[key],'Calibration changed: '+arm+'/'+name)
        fitted=read(Path(ap['assets'])/'fit.json')
        require(fitted['parameters']==ap['dynamic_parameters'],'Fit parameters differ from preregistered plan')
    output.mkdir(parents=True)
    ledger=dict(status='running',started_unix=time.time(),plan_sha256=sha(bundle/'plan.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        imported_runtime=receipt,commands=commands,arms={})
    save(output/'run_ledger.json',ledger)
    try:
        for arm in plan['run_order']:
            start=time.time()
            with (output/(arm+'.log')).open('x') as log:
                done=subprocess.run(commands[arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            ledger['arms'][arm]=dict(exit_code=done.returncode,process_seconds=time.time()-start)
            save(output/'run_ledger.json',ledger)
            require(done.returncode==0,'Failed arm; raw/partial outputs preserved')
            old_validate(read(output/(arm+'.json')),arm_plan(plan,arm),rows,'off')
        pair=combine(*[read(output/(a+'.json')) for a in plan['run_order']],plan,rows)
        pair['plan_sha256']=ledger['plan_sha256']
        pair['arm_files_sha256']={a:sha(output/(a+'.json')) for a in plan['run_order']}
        save(output/'paired.json',pair)
        cmd=[GRADER,'-u',BASE+'scripts/regrade_saved_results.py','--input',str(output/'paired.json'),
             '--output',str(output/'author_grading.json'),'--data-name','math','--group-budget-seconds','0']
        ledger['grading_command']=cmd
        save(output/'run_ledger.json',ledger)
        with (output/'author_grading.log').open('x') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        graded=read(output/'author_grading.json')
        require(graded['input_sha256']==sha(output/'paired.json'),'Grading input mismatch')
        require(graded['dataset_sha256']==plan['dataset_sha256'],'Grading dataset mismatch')
        result=analyze(pair,graded)
        result['hashes']={n:sha(output/n) for n in ('paired.json','author_grading.json')}
        save(output/'analysis.json',result)
        ledger.update(status='completed',completed_unix=time.time())
        print(json.dumps(result),flush=True)
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time()); raise
    finally:
        save(output/'run_ledger.json',ledger)


if __name__=='__main__': main()
