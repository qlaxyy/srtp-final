"""Fixed engineering checks before feedback or positive-branch screens."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mechanism_candidates import ROOT, BASE, read, save, sha, require
from run_vector_batch import validate_bundle,command,actual_parser_check,validate_arm
from run_mechanism_screen import PYTHON,runtime_check,run_child


def smoke_plan(plan,bundle):
    item=plan['engineering'];p=copy.deepcopy(plan)
    p.update(count=item['count'],dataset_file=item['dataset_file'],dataset_sha256=item['dataset_sha256'],train_indices=item['train_indices'])
    p['runtime']=dict(plan['runtime'],max_tokens=item['max_tokens'],group_timeout_seconds=item['group_timeout_seconds'])
    if plan.get('positive_branch_ablation'):
        candidate=copy.deepcopy(p['arms']['negative_only_dynamic'])
        p['arms']={'original_dynamic':p['arms']['original_dynamic'],'ablation_disabled':dict(candidate,negative_only=False),'negative_only_dynamic':candidate}
    else:
        candidate=copy.deepcopy(p['arms']['latent_feedback_clip'])
        p['arms']={'original_dynamic':p['arms']['original_dynamic'],'feedback_disabled':dict(candidate,feedback_disabled=True),'feedback_enabled':candidate}
    p['run_order']=item['order'];p['new_answers_planned']=item['new_short_outputs']
    rows=[json.loads(s) for s in (bundle/item['dataset_file']).read_text(encoding='utf-8').splitlines()]
    require(sha(bundle/item['dataset_file'])==item['dataset_sha256'],'Engineering data changed')
    require([r['train_index'] for r in rows]==item['train_indices'],'Engineering order changed')
    return p,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--runtime-check',action='store_true');a=p.parse_args()
    bundle=a.bundle.resolve();out=a.output.resolve();plan,_=validate_bundle(bundle);smoke,rows=smoke_plan(plan,bundle)
    if a.runtime_check:
        result=runtime_check(bundle,smoke,rows);result['parser']=actual_parser_check(smoke,bundle,out);print(json.dumps(result));return
    commands={name:command(smoke,bundle,out,name) for name in smoke['run_order']}
    if not a.execute:
        print(json.dumps(dict(status='cpu_preflight_passed',questions=len(rows),short_outputs=24,max_tokens=smoke['runtime']['max_tokens'],commands=commands)));return
    require(sys.platform=='linux' and not out.exists(),'Fresh Linux output required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    out.mkdir(parents=True);env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    ledger=dict(status='running',started_unix=time.time(),plan_sha256=sha(bundle/'plan.json'),commands=commands,arms={},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),purpose=plan['engineering']['purpose'])
    save(out/'ledger.json',ledger)
    try:
        preflight=[PYTHON,'-u',str(Path(__file__)),'--bundle',str(bundle),'--output',str(out),'--runtime-check']
        require(run_child(preflight,out/'runtime_check.log',env,180)==0,'Runtime parser check failed')
        if not plan.get('positive_branch_ablation'):
            kernel=[PYTHON,'-u',str(ROOT/BASE/'scripts/check_feedback_graph.py'),'--assets',str(bundle/'assets/latent_feedback_clip'),'--output',str(out/'kernel_check.json')]
            require(run_child(kernel,out/'kernel_check.log',env,600)==0,'Compiled kernel check failed')
            ledger['kernel_check']=read(out/'kernel_check.json');save(out/'ledger.json',ledger)
        results={}
        for name,cmd in commands.items():
            remaining=plan['engineering']['batch_timeout_seconds']-(time.time()-ledger['started_unix']);require(remaining>0,'Engineering deadline')
            started=time.time();code=run_child(cmd,out/(name+'.log'),env,min(600,remaining))
            ledger['arms'][name]=dict(exit_code=code,process_seconds=time.time()-started);save(out/'ledger.json',ledger)
            require(code==0,'Short generation failed; preserve partial output')
            raw=read(out/(name+'.json'));group=validate_arm(raw,smoke,rows,name)
            require(raw['provenance']['commit']==ledger['commit'],'Source changed')
            require(all(len(r['token_ids'])<=smoke['runtime']['max_tokens'] for r in group['records']),'Engineering cap changed')
            results[name]=raw;ledger['arms'][name]['raw_sha256']=sha(out/(name+'.json'));save(out/'ledger.json',ledger)
            if name in ('feedback_disabled','ablation_disabled'):
                left=results['original_dynamic'];require(raw['environment']==left['environment'],'Environment changed')
                require(all(x['token_ids']==y['token_ids'] for x,y in zip(left['rebalance_dynamic']['records'],group['records'],strict=True)),
                        'Disabled option token output is not bitwise identical; stop before enabled/screen')
        if plan.get('positive_branch_ablation'):
            enabled_stats=results['negative_only_dynamic']['rebalance_dynamic']['summary']['positive_suppression']
            require(enabled_stats['boundary_updates_cancelled']>0,'No positive cancellation observed; stop before screen')
        else:
            enabled_stats=results['feedback_enabled']['rebalance_dynamic']['summary']['feedback_applications']
        ledger.update(status='engineering_passed_no_efficacy_claim',completed_unix=time.time(),gpu_work_finished=True,
            disabled_token_pairs_identical=8,enabled_stats=enabled_stats,
            new_short_outputs=24,total_generated_tokens=sum(r['tokens'] for x in results.values() for r in x['rebalance_dynamic']['records']),
            pure_generation_seconds=sum(x['rebalance_dynamic']['summary']['generation_seconds'] for x in results.values()))
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'ledger.json',ledger)
    print(json.dumps(ledger))


if __name__=='__main__':main()
