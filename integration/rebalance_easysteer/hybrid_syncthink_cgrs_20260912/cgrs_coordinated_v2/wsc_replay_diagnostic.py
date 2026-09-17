"""Fixed unsteered calibration-prefix replay; zero new generated answers.

This is HF27 probe diagnosis, NOT reconstruction of old RC14 states. Never
loads a steering vector, alters a checkpoint, or regenerates calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_rows(ids, prompt_length, boundaries):
    previous=-1;result=[]
    for pos,token in enumerate(ids):
        if token==151649:break
        if token in boundaries:
            result.append(dict(position=pos,absolute=prompt_length+pos,
                               chunk_tokens=pos-previous-1))
            previous=pos
    return result


def chunk_targets(rows,start,end):
    return [(i,r['absolute']-start) for i,r in enumerate(rows) if start<=r['absolute']<end]


def self_test():
    r=boundary_rows([1,271,271,2,151649,271],3,{271})
    assert r==[dict(position=1,absolute=4,chunk_tokens=1),dict(position=2,absolute=5,chunk_tokens=0)]
    for size in (1,2,3,4,128,256):
        seen=[(i,start+offset) for start in range(0,9,size)
              for i,offset in chunk_targets(r,start,min(9,start+size))]
        assert seen==[(0,4),(1,5)]
    assert boundary_rows([1]*100,5,{271})==[]
    return dict(chunk_partition_position_identity=True,delimiter_excluded=True,
                answer_region_excluded=True,no_boundary_preserved=True,gpu_used=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--gpu-authorized',action='store_true')
    a=p.parse_args()
    if not a.gpu_authorized:raise ValueError('Separate bounded GPU authorization required')
    plan=json.loads(a.plan.read_text(encoding='utf8'));root=Path(__file__).resolve().parents[4]
    assert plan['kind']=='unsteered_saved_prefix_probe_diagnostic' and len(plan['cases'])==8
    assert plan['generated_tokens']==0 and plan['layer_index']==27
    assert [c['train_index'] for c in plan['cases']]==[3241,5353,759,7012,26,64,76,385]
    assert plan['partition_checks']==[5353,26]
    assert sum(len(c['token_ids'])+len(c['prompt_token_ids']) for c in plan['cases'])==14715
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
    for name,meta in plan['model_files'].items():assert sha(Path(plan['model_path'])/name)==meta['sha256'],name
    out=a.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    start=time.monotonic();completed=[]
    def timeout(*args):raise TimeoutError('600s ceiling; preserve partials, no automatic expansion')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(42)
        lm=AutoModelForCausalLM.from_pretrained(plan['model_path'],local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
        assert lm.config.num_hidden_layers==28 and lm.config.hidden_size==1536
        model=lm.model
        startup=time.monotonic()-start
        for case in plan['cases']:
            ids=case['prompt_token_ids']+case['token_ids']
            rows=boundary_rows(case['token_ids'],len(case['prompt_token_ids']),set(plan['boundary_ids']))
            for size in ([256,128] if case['train_index'] in plan['partition_checks'] else [256]):
                folder=out/f"{case['train_index']}_chunk{size}";folder.mkdir()
                hidden=np.empty((len(rows),1536),dtype=np.float32);seen=np.zeros(len(rows),dtype=bool)
                cache=None;events=[];began=time.monotonic()
                # Entire exact saved prefix is replayed, not a cropped local window.
                with torch.inference_mode():
                    for offset in range(0,len(ids),size):
                        end=min(len(ids),offset+size)
                        tokens=torch.tensor([ids[offset:end]],dtype=torch.long,device='cuda')
                        t0,t1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        t0.record()
                        result=model(input_ids=tokens,past_key_values=cache,use_cache=True,
                            output_hidden_states=True,return_dict=True)
                        t1.record();events.append((t0,t1));cache=result.past_key_values
                        assert len(result.hidden_states)==29
                        targets=chunk_targets(rows,offset,end)
                        if targets:
                            ii,jj=zip(*targets)
                            h=result.hidden_states[27][0,list(jj)].float().cpu().numpy()
                            assert np.isfinite(h).all()
                            hidden[list(ii)]=h;seen[list(ii)]=True
                        # Flush progress even if later chunks fail. No fabricated scores.
                        with (folder/'progress.json').open('w',encoding='utf8') as f:
                            json.dump(dict(processed_inputs=end,observed_boundaries=int(seen.sum())),f)
                        del result
                torch.cuda.synchronize();assert seen.all()
                seconds=sum(x.elapsed_time(y) for x,y in events)/1000
                with (folder/'features.npz').open('xb') as f:
                    np.savez(f,hidden=hidden,positions=np.array([r['position'] for r in rows]),
                             chunk_tokens=np.array([r['chunk_tokens'] for r in rows]))
                record=dict(train_index=case['train_index'],chunk_size=size,input_tokens=len(ids),
                    boundaries=len(rows),gpu_forward_seconds=seconds,wall_seconds=time.monotonic()-began)
                (folder/'result.json').write_text(json.dumps(record,indent=2));completed.append(record)
                del cache
        report=dict(completed=completed,startup_seconds=startup,wall_seconds=time.monotonic()-start,
                    generated_tokens=0,scope='Saved unsteered prefixes only; not RC14 replay or efficacy evaluation')
        (out/'complete.json').write_text(json.dumps(report,indent=2))
    except BaseException as exc:
        if 'hidden' in locals() and 'seen' in locals() and 'folder' in locals():
            try:
                with (folder/'partial_features.npz').open('xb') as f:
                    np.savez(f,hidden=hidden[seen],row_indices=np.flatnonzero(seen))
            except Exception:
                pass  # Original error and progress remain authoritative.
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),completed=completed,
            wall_seconds=time.monotonic()-start),indent=2));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
