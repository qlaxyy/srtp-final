"""One selected-layer replay for SEAL class sums; no answer generation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np
from mechanism_candidates import ROOT,read,save,sha,require
from prepare_seal_saved_replay import author_index_function,SavedTokenAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();prepared=a.prepared.resolve();out=a.output.resolve();plan=read(prepared/'replay_plan.json')
    require(plan['status']=='prepared_saved_answer_class_means_only' and plan['questions']==500,'Wrong scope')
    require(sha(prepared/'positions.json')==plan['positions_sha256'],'Positions changed')
    require(sha(prepared/'hidden_analysis_author.py',source=True)==plan['author_label_source_sha256'],'Author source changed')
    require(not out.exists(),'Output exists')
    if not a.execute:
        print(json.dumps(dict(status='cpu_preflight_passed',questions=500,new_answers=0,decoder_output_layer=19,expected_bytes=plan['expected_new_bytes'])));return
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty source')
    require(sha(Path(plan['generations']))==plan['generations_sha256'],'Saved answers changed')
    original=Path(plan['original'])
    for name,digest in plan['original_files_sha256'].items():require(sha(original/name)==digest,'Original asset changed')
    for name,digest in plan['model_files_sha256'].items():require(sha(Path(plan['model'])/name)==digest,'Model changed')
    require(shutil.disk_usage(out.parent).free>plan['expected_new_bytes']+1024**3,'Insufficient disk')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    adapter=SavedTokenAdapter(read(Path(plan['model'])/'tokenizer.json'))
    adapter.decode=lambda ids:tokenizer.decode(ids)
    author=author_index_function(prepared/'hidden_analysis_author.py')
    boundary={i for t,i in tokenizer.get_vocab().items() if 'ĊĊ' in t}
    rows=[json.loads(s) for s in Path(plan['generations']).read_text(encoding='utf-8').splitlines()];maps=read(prepared/'positions.json')
    require(len(rows)==len(maps)==500,'Missing question')
    for row,mapping in zip(rows,maps,strict=True):
        adapter.ids=row['prompt_token_ids']+row['token_ids']
        positions,checks,switches=author('saved_token_sequence',adapter,boundary,True)
        classes=[1 if i in set(checks) else 2 if i in set(switches) else 0 for i in range(len(positions))]
        require(positions==mapping['positions'] and classes==mapping['classes'],'Real-tokenizer SEAL labels differ')
    out.mkdir(parents=True);save(out/'plan.json',plan)
    ledger=dict(status='loading_model',started_unix=time.time(),new_answers=0,completed_questions=0,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),plan_sha256=sha(prepared/'replay_plan.json'),
        all_500_real_tokenizer_labels_matched=True,script_sha256=sha(Path(__file__),source=True))
    save(out/'ledger.json',ledger)
    try:
        started=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
        ledger['startup_seconds']=time.perf_counter()-started;decoder=model.model;capture={};point=None
        def selected_hook(module,inputs,value):
            h=value[0] if isinstance(value,tuple) else value
            capture['selected']=h[0,point['positions']].detach().float().cpu().numpy()
        def reference_hook(module,inputs,value):
            h=value[0] if isinstance(value,tuple) else value
            capture['reference']=h[0,point['reference_first']].detach().float().cpu().numpy()
        handles=[decoder.layers[19].register_forward_hook(selected_hook),decoder.layers[20].register_forward_hook(reference_hook)]
        sums=np.lib.format.open_memmap(out/'class_sums.npy',mode='w+',dtype=np.float64,shape=(500,3,1536))
        counts=np.lib.format.open_memmap(out/'class_counts.npy',mode='w+',dtype=np.int64,shape=(500,3))
        old=np.load(original/'layer_21.npy',mmap_mode='r');offset=0;checks=[];started=time.perf_counter()
        for q,(row,point) in enumerate(zip(rows,maps,strict=True)):
            require(time.time()-ledger['started_unix']<plan['timeout_seconds'],'Replay deadline')
            ids=row['prompt_token_ids']+row['token_ids'][:point['think_stop']]
            with torch.inference_mode():result=decoder(torch.tensor([ids],device='cuda'),use_cache=False,return_dict=True,output_hidden_states=False)
            ref=capture.pop('reference');n=len(ref);prior=np.asarray(old[offset:offset+n]);offset+=n
            relative=float(np.linalg.norm(ref.astype(float)-prior)/max(np.linalg.norm(prior.astype(float)),1e-12))
            require(relative<=.02,'Reference hidden states changed beyond fixed tolerance')
            checks.append(dict(question=q,relative_rmse=relative,all_exact=bool(np.array_equal(ref,prior))))
            h=capture.pop('selected').astype(np.float64);require(np.isfinite(h).all(),'Nonfinite states')
            classes=np.array(point['classes'])
            for value in range(3):
                mask=classes==value;sums[q,value]=h[mask].sum(0);counts[q,value]=mask.sum()
            sums.flush();counts.flush();del result
            ledger.update(status='replaying_saved_answers',completed_questions=q+1,replay_seconds=time.perf_counter()-started);save(out/'ledger.json',ledger)
            if (q+1)%50==0:print(f"SEAL class sums {q+1}/500; {ledger['replay_seconds']:.1f}s",flush=True)
        require(offset==len(old),'Incomplete reference rows')
        for handle in handles:handle.remove()
        del model,decoder;torch.cuda.empty_cache();save(out/'original_feature_checks.json',checks)
        ledger.update(status='completed',gpu_work_finished=True,completed_unix=time.time(),all_reference_states_exact=all(x['all_exact'] for x in checks),
            class_steps=counts.sum(0).tolist(),files_sha256={name:sha(out/name) for name in ['class_sums.npy','class_counts.npy','original_feature_checks.json']})
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'ledger.json',ledger)
    print(json.dumps(ledger))


if __name__=='__main__':main()
