"""Complete only the affected position-screen arms, retaining the valid control.

The first attempt's matched-first arm used evaluator default layer18 because
the new fit version did not enter its auto-code-v2 layer override. Keep that
attempt immutable. The fix is an explicit --layer20 in orchestration; actual
generation sources, original valid control and all vector assets are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from mechanism_candidates import ROOT, BASE, require, read, save, sha
from run_control_point_screen import validate_bundle, validate_arm, command, runtime_check, run_child
from run_mechanism_screen import PYTHON


def explicit_command(plan,bundle,output,name):
    cmd=command(plan,bundle,output,name)
    cmd.extend(['--layer',str(plan['decoder_output_layer'])])
    return cmd


def actual_evaluator_preflight(plan,bundle,output):
    """Exercise real parsing/spec construction; stop before LLM construction."""
    sys.path.insert(0,str(ROOT/BASE/'eval'))
    import rebalance_dynamic_eval as evaluator
    class BeforeModel(Exception): pass
    real_vector=evaluator.VectorSpec
    seen=[]
    def checked_vector(*args,**kwargs):
        require(kwargs['layers']==[plan['decoder_output_layer']], 'Actual evaluator selected wrong layer')
        seen.append(kwargs['layers'])
        return real_vector(*args,**kwargs)
    def stop_before_model(**kwargs): raise BeforeModel()
    evaluator.VectorSpec=checked_vector; evaluator.LLM=stop_before_model
    try:
        for name in plan['run_order']:
            sys.argv=[str(ROOT/BASE/'eval/rebalance_dynamic_eval.py')]+explicit_command(plan,bundle,output/'preflight_no_generation',name)[3:]
            before=len(seen)
            try:evaluator.main()
            except BeforeModel:pass
            else:raise ValueError('Evaluator preflight did not reach model constructor')
            require(len(seen)==before+1,'No checked steering spec')
        return dict(status='all_actual_evaluator_paths_checked_without_loading_model',layers=seen)
    finally:evaluator.VectorSpec=real_vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--runtime-check',action='store_true')
    a=p.parse_args();bundle=a.bundle.resolve();previous=a.previous.resolve();out=a.output.resolve()
    plan,rows=validate_bundle(bundle)
    if a.runtime_check:
        receipt=runtime_check(bundle,plan,rows)
        receipt['evaluator']=actual_evaluator_preflight(plan,bundle,out)
        print(json.dumps(receipt));return
    if not a.execute:
        print(json.dumps(dict(status='cpu_plan_passed',new_answers=200,layer=20,
            correction='Explicit layer20; no change to generation source. Retain original100, invalidate old matched100, generate matched100 and predecessor100.')));return
    require(sys.platform=='linux' and not out.exists(),'Use fresh server output')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU process')
    old=read(previous/'run_ledger.json')
    require(old['status']=='incomplete' and old['error']=="ValueError('Wrong intervention')",'Unexpected prior failure')
    require(old['plan_sha256']==sha(bundle/'plan.json'),'Old plan differs')
    control=read(previous/'original_dynamic.json')
    validate_arm(control,plan,rows,'original_dynamic')
    require(sha(previous/'original_dynamic.json')==old['arms']['original_dynamic']['raw_sha256'],'Control changed')
    bad=read(previous/'matched_first.json')
    require(bad['protocol']['easysteer_output_layer']==18,'Unexpected rejected layer')
    # The unchanged bundle validates all original runtime source hashes. Git
    # commits may differ only through orchestration/CPU/docs additions, so also
    # compare each listed source with the immutable original generation commit.
    for path,digest in plan['source_sha256'].items():
        raw=subprocess.check_output(['git','show',old['commit']+':'+path],cwd=ROOT)
        import hashlib
        require(hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==digest,'Generation source differs from retained control: '+path)
    out.mkdir(parents=True)
    for name in ['original_dynamic.json','original_dynamic.dynamic.partial.jsonl','original_dynamic.log']:
        shutil.copyfile(previous/name,out/name)
    ledger=dict(status='running',started_unix=time.time(),plan_sha256=sha(bundle/'plan.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reused_original=dict(path=str(previous/'original_dynamic.json'),sha256=sha(out/'original_dynamic.json'),
            commit=old['commit'],source_hashes_identical=True),
        rejected_attempt=dict(path=str(previous/'matched_first.json'),sha256=sha(previous/'matched_first.json'),
            reason='100 complete outputs at layer18, not planned layer20; engineering-invalid, never efficacy evidence'),
        previous_ledger_sha256=sha(previous/'run_ledger.json'),
        previous_wall_seconds=old['stopped_unix']-old['started_unix'],
        previous_matched_generation_seconds=bad['rebalance_dynamic']['summary']['generation_seconds'],
        arms={'original_dynamic':old['arms']['original_dynamic']},
        commands={n:explicit_command(plan,bundle,out,n) for n in plan['run_order'][1:]})
    save(out/'run_ledger.json',ledger)
    env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    try:
        check=[PYTHON,'-u',str(Path(__file__)),'--bundle',str(bundle),'--previous',str(previous),'--output',str(out),'--runtime-check']
        require(run_child(check,out/'runtime_check.log',env,180)==0,'Actual evaluator preflight failed')
        ledger['runtime_check']=json.loads((out/'runtime_check.log').read_text().splitlines()[-1]);save(out/'run_ledger.json',ledger)
        for name in plan['run_order'][1:]:
            remaining=1500-(time.time()-ledger['started_unix']);require(remaining>0,'Resume deadline')
            started=time.time();code=run_child(ledger['commands'][name],out/(name+'.log'),env,min(900,remaining))
            ledger['arms'][name]=dict(exit_code=code,process_seconds=time.time()-started);save(out/'run_ledger.json',ledger)
            require(code==0,'Generation failed; preserve progress')
            saved=read(out/(name+'.json'));validate_arm(saved,plan,rows,name)
            require(saved['environment']==control['environment'],'Environment differs from valid control')
            require(saved['protocol']['dynamic_params']==control['protocol']['dynamic_params'],'Controller differs')
            require(saved['provenance']['commit']==ledger['commit'],'Source changed during batch')
            ledger['arms'][name]['raw_sha256']=sha(out/(name+'.json'));save(out/'run_ledger.json',ledger)
        ledger.update(status='generation_completed_grading_pending',completed_unix=time.time(),gpu_work_finished=True)
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'run_ledger.json',ledger)


if __name__=='__main__':main()
