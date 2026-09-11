"""Bounded selected-layer teacher-forced replay; never generates calibration answers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import numpy as np
from mechanism_candidates import ROOT, read, save, sha, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); prepared=a.prepared.resolve(); output=a.output.resolve()
    plan=read(prepared/'replay_plan.json')
    require(plan['status']=='prepared_replay_saved_answers_only','Wrong replay scope')
    require(sha(prepared/'positions.json')==plan['mapping_sha256'],'Mapping changed')
    require(sha(prepared/'matched_mask.npy')==plan['matched_mask_sha256'],'Mask changed')
    require(not output.exists(),'Replay output exists; preserve completed/partial files')
    if not a.execute:
        print(json.dumps(dict(status='cpu_preflight_passed_no_model_import',plan_sha256=sha(prepared/'replay_plan.json'),questions=plan['questions'],new_answers=0)))
        return
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    require(not processes,'Other GPU process exists')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty checkout')
    original=Path(plan['original'])
    for name,digest in plan['original_files_sha256'].items(): require(sha(original/name)==digest,'Original asset changed: '+name)
    require(sha(Path(plan['generations']))==plan['generations_sha256'],'Generations changed')
    for name,digest in plan['model_files_sha256'].items(): require(sha(Path(plan['model'])/name)==digest,'Model changed: '+name)
    require(shutil.disk_usage(output.parent).free>plan['expected_new_bytes']+1024**3,'Insufficient disk')
    maps=read(prepared/'positions.json')
    rows=[json.loads(s) for s in Path(plan['generations']).read_text(encoding='utf-8').splitlines()]
    require(len(rows)==len(maps)==500,'Wrong number of questions')
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    output.mkdir(parents=True)
    save(output/'plan.json',plan)
    ledger=dict(status='loading_model',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                plan_sha256=sha(prepared/'replay_plan.json'),started_unix=time.time(),new_answers=0,completed_questions=0,
                script_sha256=sha(Path(__file__),source=True),torch=torch.__version__)
    save(output/'ledger.json',ledger)
    try:
        started=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval()
        ledger['startup_seconds']=time.perf_counter()-started
        decoder=model.model; capture={}; point=None
        def hook(module,inputs,value):
            h=value[0] if isinstance(value,tuple) else value
            capture['predecessor']=h[0,point['predecessor']].detach().float().cpu().numpy()
            capture['first']=h[0,point['first']].detach().float().cpu().numpy()
            capture['prompt']=h[0,:point['prompt_tokens']].float().mean(0).detach().cpu().numpy()
        handle=decoder.layers[plan['decoder_output_layer']].register_forward_hook(hook)
        x=np.lib.format.open_memmap(output/'predecessor_layer21.npy',mode='w+',dtype=np.float32,shape=(plan['steps'],1536))
        prompts=np.lib.format.open_memmap(output/'prompt_mean_layer21.npy',mode='w+',dtype=np.float32,shape=(500,1536))
        old=np.load(original/'layer_21.npy',mmap_mode='r')
        offset=0; comparisons=[]; started=time.perf_counter(); input_tokens=0
        for q,(row,point) in enumerate(zip(rows,maps)):
            require(time.time()-ledger['started_unix']<plan['timeout_seconds'],'Replay time limit reached')
            ids=row['prompt_token_ids']+row['token_ids'][:point['think_stop']]
            input_tokens+=len(ids)
            with torch.inference_mode():
                result=decoder(torch.tensor([ids],device='cuda'),use_cache=False,return_dict=True,output_hidden_states=False)
            n=len(point['first']); ref=np.asarray(old[offset:offset+n],dtype=np.float64)
            actual=capture.pop('first').astype(np.float64)
            relative=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-12))
            comparisons.append(dict(question=q,positions=n,relative_rmse=relative,exact_elements=int(np.count_nonzero(actual==ref)),elements=int(ref.size)))
            require(relative<=.02,'Original first-content states differ beyond fixed tolerance')
            require(all(np.isfinite(v).all() for v in capture.values()),'Nonfinite hidden states')
            x[offset:offset+n]=capture.pop('predecessor');prompts[q]=capture.pop('prompt');offset+=n
            x.flush();prompts.flush();del result
            ledger.update(status='replaying_saved_answers',completed_questions=q+1,completed_steps=offset,replay_seconds=time.perf_counter()-started,input_tokens=input_tokens)
            save(output/'ledger.json',ledger)
            if (q+1)%25==0: print(f"Replay {q+1}/500; {ledger['replay_seconds']:.1f}s",flush=True)
        require(offset==plan['steps'],'Incomplete feature rows')
        handle.remove();del model,decoder
        torch.cuda.empty_cache()
        save(output/'original_feature_checks.json',comparisons)
        ledger.update(status='completed',completed_unix=time.time(),gpu_work_finished=True,
                      files_sha256={n:sha(output/n) for n in ['predecessor_layer21.npy','prompt_mean_layer21.npy','original_feature_checks.json']},
                      max_original_relative_rmse=max(x['relative_rmse'] for x in comparisons),
                      all_original_elements_exact=all(x['exact_elements']==x['elements'] for x in comparisons))
        save(output/'ledger.json',ledger);print(json.dumps(ledger),flush=True)
    except BaseException as e:
        ledger.update(status='incomplete',error=repr(e),stopped_unix=time.time())
        save(output/'ledger.json',ledger);raise


if __name__=='__main__':main()
