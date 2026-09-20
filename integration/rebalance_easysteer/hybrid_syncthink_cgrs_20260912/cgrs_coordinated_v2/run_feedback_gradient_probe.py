"""Full saved-response fixed-history gradient engineering, never optimizer steps."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    plan=json.loads((a.input/'plan.json').read_text());a.output.mkdir(exist_ok=False,parents=True)
    start=time.monotonic()
    def timeout(*_): raise TimeoutError('900s fixed gradient engineering ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan.get('process_ceiling_seconds',900))
    try:
        for name,h in plan['input_sha256'].items(): assert sha(a.input/name)==h,name
        replay=Path(plan['replay_directory']); complete=json.loads((replay/'complete.json').read_text())
        assert (complete.get('history_exact') or complete.get('native_history_recorded')) and complete['raw_confidence_exact']
        assert sha(replay/'complete.json')==plan['replay_complete_sha256']
        manifest=json.loads((replay/'scoring_capture_manifest.json').read_text())
        for r in manifest: assert sha(replay/r['file'])==r['sha256']
        assets=plan['assets']
        for n,item in assets['model_files'].items(): assert sha(Path(assets['model_path'])/n)==item['sha256'],n
        assert sha(assets['vector']['path'])==assets['vector']['sha256']
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from feedback_fixed_history import FixedHistoryScorer
        if plan.get('query_chunk'):
            from feedback_chunked_attention import install
            install(plan['query_chunk'])
        torch.manual_seed(42)
        model=AutoModelForCausalLM.from_pretrained(assets['model_path'],torch_dtype=getattr(torch,plan.get('hf_dtype','bfloat16')),
            attn_implementation=plan.get('hf_attention','sdpa'),local_files_only=True).to('cuda').eval()
        for p in model.parameters(): p.requires_grad_(False)
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        boundaries=torch.tensor(sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s),device='cuda')
        vec=torch.load(assets['vector']['path'],map_location='cuda',weights_only=True).float()
        tables={k:torch.as_tensor(v,device='cuda') for k,v in dict(np.load(a.input/'opening.npz')).items()}
        tables['candidate_ids']=tables['candidate_ids'].long()
        scorer=FixedHistoryScorer(model,vec,boundaries,tables)
        rows=json.loads((a.input/'rows.json').read_text()); reports=[]
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            key=row['key'];trace=dict(np.load(replay/('scoring_capture_'+key+'.npz')))
            torch.cuda.synchronize();t0=time.monotonic()
            with torch.no_grad():
                reference=scorer.score(row['prompt_token_ids'],row['token_ids'],trace).detach()
            delta=torch.zeros_like(vec,requires_grad=True)
            values=scorer.score(row['prompt_token_ids'],row['token_ids'],trace,delta)
            assert torch.equal(values.detach(),reference),(key,'zero residual changed HF reference')
            values.sum().backward()
            assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.norm()>0,(key,'gradient invalid')
            assert all(p.grad is None for p in model.parameters()),'Model weights received gradients'
            finite_difference = None
            if plan.get('finite_difference_epsilon'):
                eps = plan['finite_difference_epsilon']
                step = eps * delta.grad.detach() / delta.grad.norm()
                with torch.no_grad():
                    shifted = scorer.score(row['prompt_token_ids'],row['token_ids'],trace,step)
                actual = float((shifted.double()-reference.double()).sum())
                predicted = float((step.double()*delta.grad.double()).sum())
                finite_difference = dict(epsilon=eps,actual_logp_change=actual,predicted_logp_change=predicted,
                    relative_error=abs(actual-predicted)/abs(predicted))
                del shifted
            torch.cuda.synchronize();elapsed=time.monotonic()-t0
            ref=reference.cpu().numpy();grad=delta.grad.detach().cpu().numpy()
            name=key+'.npz';np.savez_compressed(a.output/name,logp=ref,gradient=grad)
            difference=ref-trace['logp']
            reports.append(dict(key=key,question=row['question'],chosen=row['chosen'],source=row['source'],
                response_tokens=len(row['token_ids']),sequence_logp=float(reference.sum()),
                gradient_norm=float(delta.grad.norm()),zero_residual_exact=True,
                native_vllm_vs_fixed_hf_mean_abs_logp=float(np.abs(difference).mean()),
                native_vllm_vs_fixed_hf_max_abs_logp=float(np.abs(difference).max()),
                finite_difference=finite_difference,seconds=elapsed,file=name,sha256=sha(a.output/name),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            save(a.output/'progress.json',reports)
            print(key,'complete',elapsed,flush=True)
            del values,reference,delta
        scorer.close()
        save(a.output/'complete.json',dict(status='Gradient engineering complete, NO optimization or efficacy result',
            model_forward_count=(3 if plan.get('finite_difference_epsilon') else 2)*len(rows),backward_count=len(rows),new_answer_count=0,optimizer_steps=0,
            rows=reports,wall_seconds=time.monotonic()-start,
            limits=['Fixed native histories and lexical masks, not updated online policy',
                    'HF is a numerical surrogate, not exact vLLM replay; score discrepancies recorded',
                    'Nonzero gradients do not establish helpful direction or compression',
                    'Autograd and finite differences describe only this configured fixed-history surrogate'],
            plan_sha256=sha(a.input/'plan.json')))
    except BaseException:
        save(a.output/'failure.json',dict(error=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
