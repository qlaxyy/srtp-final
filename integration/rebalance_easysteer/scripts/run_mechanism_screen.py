"""Default: CPU preflight. --execute: only the explicitly authorized fixed GPU batch."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time

from mechanism_candidates import ROOT, BASE, require, read, save, sha, read_vector

PYTHON='/root/autodl-tmp/venvs/easysteer-vllm026/bin/python'
GRADER='/root/autodl-tmp/venvs/rebalance/bin/python'


def validate_bundle(bundle,check_source=True):
    plan=read(bundle/'plan.json')
    require(plan['status']=='prepared_not_run' and plan['count']==100,'Not the fixed screen')
    require(plan['run_order']==['original_dynamic','min_displacement','orthogonal_mean'],'Unexpected scope')
    require(plan['new_answers_planned']==300 and plan['runtime']['max_tokens']==16000,'Wrong count/cap')
    require(sha(bundle/'screen100.jsonl')==plan['dataset_sha256'],'Screen dataset changed')
    require(sha(bundle/'confirmation200.jsonl')==plan['confirmation']['dataset_sha256'],'Reserve changed')
    rows=[json.loads(s) for s in (bundle/'screen100.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows)==100 and [r['train_index'] for r in rows]==plan['train_indices'],'Screen order changed')
    require(not any(plan['overlap_checks'].values()),'Dataset overlap')
    original=bundle/plan['arms']['original_dynamic']['directory']/'auto_vector.pt'
    for name,arm in plan['arms'].items():
        folder=bundle/arm['directory'];fit=read(folder/'fit.json')
        require(sha(folder/'auto_vector.pt')==arm['vector_sha256']==fit['vector_sha256'],'Vector changed: '+name)
        require(sha(folder/'fit.json')==arm['fit_sha256'],'Fit changed: '+name)
        require(fit['parameters']==plan['dynamic_parameters'],'Controller changed: '+name)
        require(fit['decoder_output_layer']==plan['decoder_output_layer'],'Layer changed')
        read_vector(folder/'auto_vector.pt',original)
    if check_source:
        for name,expected in plan['source_sha256'].items():
            require(sha(ROOT/name,source=True)==expected,'Source changed: '+name)
    return plan,rows


def command(plan,bundle,output,arm,root=ROOT):
    folder=bundle/plan['arms'][arm]['directory']
    cmd=[PYTHON,'-u',str(root/BASE/'eval/rebalance_dynamic_eval.py'),'--model',plan['model'],
         '--dataset',str(bundle/'screen100.jsonl'),'--limit','100','--vector',str(folder/'auto_vector.pt'),
         '--calibration-fit',str(folder/'fit.json'),'--diagnostic-group','rebalance_dynamic','--output',str(output/(arm+'.json'))]
    for key,value in plan['runtime'].items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:cmd.append(flag)
        else:cmd.extend([flag,str(value)])
    return cmd


def validate_arm(saved,plan,rows,arm):
    require(saved.get('status')=='diagnostic_completed','Incomplete arm; preserve partial answers')
    require('baseline' not in saved,'Unexpected unsteered generation')
    p=saved['protocol'];asset=plan['arms'][arm]
    for key,value in plan['runtime'].items():require(p[key]==value,'Runtime mismatch: '+key)
    require(p['offset']==0 and p['limit']==100 and p['model']==plan['model'],'Wrong model/slice')
    require(p['run_order']==['rebalance_dynamic'] and p['easysteer_output_layer']==20,'Wrong intervention')
    require(not p.get('profiling_enabled') and not p.get('repeat_gate'),'Unplanned observer')
    for key,value in plan['dynamic_parameters'].items():require(p['dynamic_params'][key]==value,'Controller mismatch')
    require(not p['dynamic_params'].get('inject_first_step',False),'Unplanned first-prompt injection')
    for key,expected in [('dataset_sha256',plan['dataset_sha256']),('vector_sha256',asset['vector_sha256']),('calibration_fit_sha256',asset['fit_sha256'])]:
        require(saved['provenance'][key]==expected,'Asset mismatch: '+key)
    require(not saved['provenance']['git_status'].strip(),'Dirty generation worktree')
    group=saved['rebalance_dynamic'];records=group['records']
    require(len(records)==len(rows)==100,'Missing records')
    for i,(record,row) in enumerate(zip(records,rows,strict=True)):
        require(record['dataset_index']==i and record['problem']==row['problem'] and record['gold']==row['answer'],'Question pairing mismatch')
        require(record['tokens']==len(record['token_ids'])<=16000,'Invalid token count')
        require(0<=record['thinking_tokens']<=record['tokens'],'Invalid thinking count')
        require(record['finish_reason'] in ('stop','length'),'Unfinished answer')
    require(group['summary']['generation_seconds']>0,'Missing generation timing')
    return group


def runtime_check(bundle,plan,rows):
    """Runs in the existing server runtime, before any model is loaded."""
    import inspect
    import numpy as np
    import torch
    import vllm
    from transformers import AutoTokenizer
    from easysteer.vectors import from_pt_direction
    require(torch.cuda.is_available(),'Requested GPU mode unavailable; do not reinstall')
    require(Path(vllm.__file__).resolve()==(ROOT/'sources/EasySteer/vllm-steer/vllm/__init__.py').resolve(),'vLLM editable path differs; stop without reinstalling')
    require(Path(inspect.getfile(from_pt_direction)).resolve()==(ROOT/'sources/EasySteer/easysteer/vectors.py').resolve(),'EasySteer editable path differs')
    for name,expected in plan['model_files_sha256'].items():require(sha(Path(plan['model'])/name)==expected,'Model asset changed: '+name)
    original=bundle/plan['arms']['original_dynamic']['directory']/'auto_vector.pt'
    for arm in plan['arms'].values():
        path=bundle/arm['directory']/'auto_vector.pt'
        value=torch.load(path,map_location='cpu',weights_only=True)
        require(value.shape==(1536,) and value.dtype==torch.float32 and torch.isfinite(value).all().item(),'Invalid tensor')
        require(np.array_equal(value.numpy(),read_vector(path,original)),'Torch storage differs from CPU checked asset')
        from_pt_direction(str(path),layers=[20]).to_wire()
    sys.path.insert(0,str(ROOT/BASE/'eval'))
    from rebalance_static_eval import build_prompt
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    max_prompt=max(len(tokenizer.encode(build_prompt(tokenizer,row['problem']))) for row in rows)
    require(max_prompt+16000<=32768,'Actual prompt context overflow')
    return dict(status='server_runtime_and_tensors_checked_before_model_load',max_prompt_tokens=max_prompt,
                torch=torch.__version__,vllm=vllm.__version__,cuda=torch.version.cuda,
                vllm_file=vllm.__file__,vector_file=inspect.getfile(from_pt_direction))


def run_child(cmd,log_path,env,timeout):
    """Bound only this child's process group; keep completed and partial files."""
    with log_path.open('x',encoding='utf-8') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--runtime-check',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();bundle=args.bundle.resolve();output=args.output.resolve()
    plan,rows=validate_bundle(bundle)
    if args.runtime_check:
        require(sys.platform=='linux','Runtime acceptance uses the existing server only')
        print(json.dumps(runtime_check(bundle,plan,rows)));return
    commands={name:command(plan,bundle,output,name) for name in plan['run_order']}
    if not args.execute:
        server_root=PurePosixPath('/root/autodl-tmp/projects/srtp-final')
        server_bundle=server_root/bundle.relative_to(ROOT).as_posix()
        print(json.dumps(dict(status='local_cpu_preflight_passed_no_generation',count=100,new_answers=300,
            run_order=plan['run_order'],gpu_authorized=False,local_asset_hashes_checked=True,
            pending='Live server/model hashes, actual tokenizer and torch/EasySteer loading before model startup',
            server_commands={name:command(plan,server_bundle,PurePosixPath(plan['default_server_output']),name,server_root) for name in plan['run_order']}),indent=2));return
    require(sys.platform=='linux','GPU batch uses the existing Linux server')
    require(not output.exists(),'Fresh output directory required; never auto-rerun completed arms')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit worktree first')
    require(Path(PYTHON).is_file() and Path(GRADER).is_file(),'Existing environments missing; stop without reinstalling')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    require(not active,'GPU has another compute process; inspect without disturbing it: '+active)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    output.mkdir(parents=True)
    env=dict(os.environ,PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    started=time.time();ledger=dict(status='running',started_unix=started,gpu=gpu,plan_sha256=sha(bundle/'plan.json'),commands=commands,arms={},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(output/'run_ledger.json',ledger)
    try:
        check=[PYTHON,'-u',str(Path(__file__).resolve()),'--bundle',str(bundle),'--output',str(output),'--runtime-check']
        code=run_child(check,output/'runtime_check.log',env,180)
        require(code==0,'Runtime preflight failed; preserve receipt and stop')
        ledger['runtime_check']=read_last_json(output/'runtime_check.log');save(output/'run_ledger.json',ledger)
        first=None
        for name in plan['run_order']:
            remaining=1800-(time.time()-started)
            require(remaining>0,'Batch deadline reached; no new arm launched')
            arm_started=time.time();code=run_child(commands[name],output/(name+'.log'),env,min(900,remaining))
            ledger['arms'][name]=dict(exit_code=code,process_seconds=time.time()-arm_started)
            save(output/'run_ledger.json',ledger)
            require(code==0,'Generation failed; completed and partial answers preserved')
            saved=read(output/(name+'.json'));validate_arm(saved,plan,rows,name)
            if first is None:first=saved
            else:
                require(saved['environment']==first['environment'],'Runtime environment changed between arms')
                require(saved['provenance']['commit']==first['provenance']['commit']==ledger['commit'],'Code changed between arms')
                require(saved['protocol']['dynamic_params']==first['protocol']['dynamic_params'],'Boundary IDs or controller differ')
            ledger['arms'][name]['raw_sha256']=sha(output/(name+'.json'));save(output/'run_ledger.json',ledger)
        ledger.update(status='generation_completed_grading_pending',completed_unix=time.time(),gpu_work_finished=True)
        print('GPU generation finished. All model processes exited. GPU may be shut down; preserve and download outputs. Author grading is a separate CPU-only command.',flush=True)
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(output/'run_ledger.json',ledger)


def read_last_json(path):
    return json.loads(path.read_text(encoding='utf-8').splitlines()[-1])


if __name__=='__main__':main()
