"""Complete only the missing enabled smoke under a newly controlled design."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mechanism_candidates import ROOT,BASE,read,save,sha,require
from run_vector_batch import validate_bundle,command,actual_parser_check,validate_arm
from run_feedback_engineering import smoke_plan
from run_mechanism_screen import PYTHON,runtime_check,run_child


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--runtime-check',action='store_true');a=p.parse_args()
    bundle=a.bundle.resolve();out=a.output.resolve();plan,_=validate_bundle(bundle);smoke,rows=smoke_plan(plan,bundle)
    require(plan['graph_control_comparison'] and smoke['run_order']==['feedback_enabled'],'Wrong design')
    if a.runtime_check:
        r=runtime_check(bundle,smoke,rows);r['parser']=actual_parser_check(smoke,bundle,out);print(json.dumps(r));return
    cmd=command(smoke,bundle,out,'feedback_enabled')
    if not a.execute:print(json.dumps(dict(status='cpu_preflight_passed',new_short_outputs=8,command=cmd)));return
    require(sys.platform=='linux' and not out.exists(),'Fresh server output required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    prior=Path(plan['engineering']['reuse_directory'])
    for name,digest in plan['engineering']['reuse_sha256'].items():require(sha(prior/name)==digest,'Short control changed')
    for name,digest in plan['engineering']['core_sources_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Core changed since short control')
    controls={name:read(prior/(name+'.json')) for name in ['original_dynamic','feedback_disabled']}
    for name,raw in controls.items():validate_arm(raw,smoke,rows,name)
    require(read(prior/'kernel_check.json')['status']=='compiled_and_cudagraph_checks_passed','Missing kernel check')
    out.mkdir(parents=True);env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0');env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    ledger=dict(status='running',plan_sha256=sha(bundle/'plan.json'),started_unix=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reused_short_groups_sha256=plan['engineering']['reuse_sha256'],original_compiled_off_identity_passed=False,purpose=plan['engineering']['purpose'])
    save(out/'ledger.json',ledger)
    try:
        preflight=[PYTHON,'-u',str(Path(__file__)),'--bundle',str(bundle),'--output',str(out),'--runtime-check']
        require(run_child(preflight,out/'runtime_check.log',env,180)==0,'Parser/runtime failed')
        started=time.time();code=run_child(cmd,out/'feedback_enabled.log',env,480)
        ledger.update(exit_code=code,process_seconds=time.time()-started);save(out/'ledger.json',ledger)
        require(code==0,'Enabled smoke incomplete')
        raw=read(out/'feedback_enabled.json');group=validate_arm(raw,smoke,rows,'feedback_enabled')
        require(raw['provenance']['commit']==ledger['commit'],'Code changed')
        require(all(raw['environment']==r['environment'] for r in controls.values()),'Environment changed')
        stats=group['summary']['feedback_applications'];require(stats['coefficient_changed']>0,'No real clipping exercised')
        require(all(r['tokens']<=256 for r in group['records']),'Wrong smoke cap')
        ledger.update(status='engineering_passed_no_efficacy_claim',completed_unix=time.time(),gpu_work_finished=True,new_short_outputs=8,
            pure_generation_seconds=group['summary']['generation_seconds'],stats=stats,raw_sha256=sha(out/'feedback_enabled.json'))
    except BaseException as error:ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'ledger.json',ledger)
    print(json.dumps(ledger))


if __name__=='__main__':main()
