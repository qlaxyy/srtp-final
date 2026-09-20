"""Independent bounded teacher-forced HF forward, U only; no generation."""
import argparse,json,time
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    import torch
    from transformers import AutoModelForCausalLM
    start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    rows=[json.loads(s) for s in a.replay.read_text().splitlines()];results=[]
    with torch.inference_mode():
        for r in rows:
            prompt=r['prompt_token_ids'];n=32
            inputs=torch.tensor([prompt+r['token_ids'][:n-1]],device='cuda')
            logits=model(input_ids=inputs,use_cache=False).logits[0,len(prompt)-1:len(prompt)-1+n].float()
            prob=(logits.amax(-1)-logits.logsumexp(-1)).exp().cpu().numpy()
            ref=np.exp(r['logmax'][:n]);diff=np.abs(prob-ref)
            # Distinct BF16 implementations: bounded tolerance fixed before run.
            results.append(dict(train_index=r['train_index'],n=n,max_abs_probability_error=float(diff.max()),
                mean_abs_probability_error=float(diff.mean()),aligned_mae=float(diff.mean()),
                shifted_mae=float(np.abs(prob[:-1]-ref[1:]).mean()),passed=bool(diff.max()<=.03)))
    a.output.write_text(json.dumps(dict(passed=all(r['passed'] and r['aligned_mae']<r['shifted_mae'] for r in results),
        results=results,seconds=time.monotonic()-start,tolerance=.03,
        interpretation='Cross-engine probability alignment tolerance, not bit-exact reproduction of historical generation.'),indent=2))

if __name__=='__main__':main()
