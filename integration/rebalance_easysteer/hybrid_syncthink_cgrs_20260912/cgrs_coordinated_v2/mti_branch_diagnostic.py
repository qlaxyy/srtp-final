"""Bounded stage-0 GPU oracle: eight saved prefixes, no generated answers.

This intentionally uses HF, no steering and no native paged cache. Passing
does NOT authorize a full evaluation or certify the vLLM production adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def cache_hash(cache):
    h=hashlib.sha256()
    # Deliberately reject unknown cache representations rather than skip checks.
    if hasattr(cache,'to_legacy_cache'):
        pairs=cache.to_legacy_cache()
    elif hasattr(cache,'layers'):
        pairs=[(layer.keys,layer.values) for layer in cache.layers]
    else: raise TypeError('Unsupported cache fingerprint format')
    for pair in pairs:
        for tensor in pair:
            h.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
            h.update(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes())
    h.update(str(cache.get_seq_length()).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-authorized',action='store_true')
    a=p.parse_args()
    if not a.gpu_authorized: raise ValueError('GPU execution must be explicitly enabled')
    plan=json.loads(a.plan.read_text(encoding='utf8'))
    precision_diagnostic=plan['kind']=='mti_stage0_fp32_failed_case_diagnostic'
    assert plan['kind'] in ('mti_stage0_unsteered_cache_oracle','mti_stage0_fp32_failed_case_diagnostic')
    assert len(plan['cases'])==(1 if precision_diagnostic else 8) and plan['generated_tokens']==0
    if precision_diagnostic: assert plan['cases'][0]['train_index']==3241
    out=a.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    records=[];start=time.monotonic()
    def stop(*args):raise TimeoutError('600s hard stop, no expansion')
    signal.signal(signal.SIGALRM,stop);signal.alarm(600)
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
        from mti_branch import isolated_cue
        for name,digest in plan['source_sha256_lf'].items():
            actual=hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            assert actual==digest,name
        for name,meta in plan['model_files'].items():
            assert sha(Path(plan['model_path'])/name)==meta['sha256'],name
        torch.manual_seed(42)
        if precision_diagnostic:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
        tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True)
        cue_ids=tokenizer.encode(plan['cue'],add_special_tokens=False)
        assert cue_ids and len(cue_ids)<=16
        lm=AutoModelForCausalLM.from_pretrained(plan['model_path'],local_files_only=True,
                    torch_dtype=torch.float32 if precision_diagnostic else torch.bfloat16,
                    attn_implementation='sdpa').eval().to('cuda')
        assert lm.config.num_hidden_layers==28 and lm.config.hidden_size==1536
        decoder=lm.model;cue=torch.tensor([cue_ids],device='cuda')
        startup=time.monotonic()-start
        def forward(ids,cache=None):
            if cache is None: cache=DynamicCache()
            result=decoder(input_ids=ids,past_key_values=cache,use_cache=True,return_dict=True)
            return result
        def probs(hidden):return lm.lm_head(hidden[:,-1,:]).float().softmax(-1)
        def timed(fn):
            torch.cuda.synchronize();before=time.monotonic();value=fn();torch.cuda.synchronize()
            return value,time.monotonic()-before
        with torch.inference_mode():
            for case in plan['cases']:
                ids=torch.tensor([case['prefix_ids']],device='cuda')
                prefix,tpre=timed(lambda:forward(ids))
                cache=prefix.past_key_values
                before_hash=cache_hash(cache)
                cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                branch,tbranch=timed(lambda:isolated_cue(decoder,cache,ids.shape[1],cue))
                rng_ok=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                after_hash=cache_hash(cache)
                ref,tref=timed(lambda:forward(torch.cat((ids,cue),1)))
                bp=probs(branch);rp=probs(ref.last_hidden_state)
                error=(bp-rp).abs().max().item();same=bp.argmax(-1).item()==rp.argmax(-1).item()
                # Continue the parent with an existing saved token, never sample.
                nxt=torch.tensor([[case['next_saved_token']]],device='cuda')
                continuation,tcont=timed(lambda:forward(nxt,cache))
                contref,tcontref=timed(lambda:forward(torch.cat((ids,nxt),1)))
                cp=probs(continuation.last_hidden_state);crp=probs(contref.last_hidden_state)
                cont_error=(cp-crp).abs().max().item();cont_same=cp.argmax(-1).item()==crp.argmax(-1).item()
                record=dict(train_index=case['train_index'],prefix_tokens=ids.shape[1],cue_tokens=len(cue_ids),
                    branch_max_probability_error=error,branch_top1_equal=same,
                    parent_continuation_probability_error=cont_error,parent_continuation_top1_equal=cont_same,
                    parent_cache_sha_before=before_hash,parent_cache_sha_after=after_hash,rng_unchanged=rng_ok,
                    seconds=dict(prefix=tpre,branch_including_copy=tbranch,full_cue_reference=tref,
                                 parent_continuation=tcont,full_continuation_reference=tcontref))
                record['passed']=bool(before_hash==after_hash and rng_ok and same and cont_same
                    and error<=plan['max_probability_error'] and cont_error<=plan['max_probability_error'])
                records.append(record)
                with (out/'records.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
                if not record['passed']:raise RuntimeError('Stage0 equivalence gate failed; stop')
                del prefix,cache,ref,continuation,contref,branch
        result=dict(stage0_passed=not precision_diagnostic,precision_diagnostic_passed=precision_diagnostic,
            cases=len(records),generated_tokens=0,steering=False,
            startup_seconds=startup,wall_seconds=time.monotonic()-start,cue_ids=cue_ids,
            forward_calls=5*len(records),native_vllm_validated=False,full_evaluation_allowed=False,
            next='Implement native paged branch plus dynamic steering-history equivalence separately')
        (out/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),
            completed_cases=len(records),wall_seconds=time.monotonic()-start),indent=2),encoding='utf8')
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
