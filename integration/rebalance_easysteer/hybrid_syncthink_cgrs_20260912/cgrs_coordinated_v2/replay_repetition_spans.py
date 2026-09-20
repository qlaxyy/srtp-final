"""GPU measurement only: saved-token replay, no generation or online controller changes."""
import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((a.input/'plan.json').read_text());a.output.mkdir(exist_ok=False,parents=True)
    def fail_timeout(*_):raise TimeoutError('Fixed600s measurement limit')
    signal.signal(signal.SIGALRM,fail_timeout);signal.alarm(600)
    started=time.monotonic()
    try:
        for n,h in plan['input_sha256'].items():assert hashlib.sha256((a.input/n).read_bytes()).hexdigest()==h,n
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['script_sha256']
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(42)
        model=AutoModelForCausalLM.from_pretrained(plan['model_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        expected=np.load(a.input/'expected.npz');rows=json.loads((a.input/'rows.json').read_text());vectors=[];receipts=[];forward=0.;capture={};positions=None
        def hook(_,args,output):
            z=output[0] if isinstance(output,tuple) else output
            for name,pos in positions.items():
                capture[name]=z[0,pos].float().mean(0).detach().cpu().numpy()
        handle=model.model.layers[20].register_forward_hook(hook)
        with torch.inference_mode():
            for r in rows:
                q=r['question'];offset=len(r['prompt_token_ids']);positions={k:[offset+i for i in v] for k,v in r['positions'].items()}
                ids=torch.tensor([r['prompt_token_ids']+r['token_ids']],device='cuda')
                torch.cuda.synchronize();begin=time.monotonic();model.model(input_ids=ids,use_cache=False,return_dict=True);torch.cuda.synchronize();forward+=time.monotonic()-begin
                for role in ['earlier','repeat']:
                    assert np.array_equal(capture[role+'_first'],expected[str(q)+'_'+role]),(q,role,'old first-state mismatch')
                row={k:v for k,v in capture.items()};assert all(np.isfinite(v).all() and v.shape==(1536,) for v in row.values())
                name=str(q)+'.npz';np.savez_compressed(a.output/name,**row);vectors.append({k:v.tolist() for k,v in row.items()})
                receipts.append(dict(question=q,file=name,sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest(),first_states_exact=True))
                capture.clear()
        handle.remove()
        (a.output/'manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
        (a.output/'complete.json').write_text(json.dumps(dict(status='complete measurement only',questions=len(rows),old_first_states_exact=True,forward_seconds=forward,wall_seconds=time.monotonic()-started,new_generation_count=0,model_forward_count=len(rows),plan_sha256=hashlib.sha256((a.input/'plan.json').read_bytes()).hexdigest()),indent=2)+'\n')
    except BaseException as e:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(e),wall_seconds=time.monotonic()-started),indent=2));raise


if __name__=='__main__':main()
