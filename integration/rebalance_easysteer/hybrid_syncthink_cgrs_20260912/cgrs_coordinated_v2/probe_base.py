"""Base model prefix scoring isolation; three one-token probes, no training."""
import argparse, json, hashlib, os, sys, time, signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=['hf','native'],required=True);a=p.parse_args()
a.output.mkdir(exist_ok=False);start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s limit')));signal.alarm(300)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 plan=json.loads((a.input/'plan.json').read_text());assert sha(a.input/'probe_base.py')==plan['script_sha256']
 original=Path(plan['original']);assert sha(original/'plan.json')==plan['original_plan_sha256']
 base=json.loads((original/'plan.json').read_text());assets=base['assets']
 for n,h in base['input_sha256'].items():assert sha(original/n)==h,n
 for n,v in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==v['sha256'],n
 rows={r['key']:r for r in json.loads((original/'rows.json').read_text())}
 selected=[(x,rows[x['key']]) for x in plan['targets']]
 prompts=[r['prompt_token_ids']+r['token_ids'][:x['index']] for x,r in selected]
 reports=[]
 if a.engine=='native':
  runtime=Path(plan['runtime']);os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
  sys.path[1:1]=[str(runtime/'sources/EasySteer/vllm-steer'),str(runtime/'sources/EasySteer')]
  from vllm import LLM,SamplingParams
  m=LLM(model=assets['model_path'],dtype='bfloat16',max_model_len=16384,max_num_seqs=16,max_num_batched_tokens=32768,gpu_memory_utilization=.9,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42,max_logprobs=-1)
  outputs=m.generate([dict(prompt_token_ids=t) for t in prompts],SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,logprobs=-1,seed=42),use_tqdm=False)
  for (x,r),o in zip(selected,outputs):
   z=o.outputs[0].logprobs[0];target=r['token_ids'][x['index']];top=sorted(z,key=lambda k:z[k].logprob,reverse=True)[:5]
   reports.append(dict(key=x['key'],target_id=target,target_logp=z[target].logprob,top5_ids=top,top5_logp=[z[k].logprob for k in top],full_vocab_count=len(z)))
 else:
  import torch
  from transformers import AutoModelForCausalLM
  m=AutoModelForCausalLM.from_pretrained(assets['model_path'],torch_dtype=getattr(torch,plan.get('hf_dtype','bfloat16')),attn_implementation=plan.get('hf_attention','sdpa'),local_files_only=True).cuda().eval()
  with torch.inference_mode():
   for (x,r),ids in zip(selected,prompts):
    h=m.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).last_hidden_state[0,-1];z=m.lm_head(h).float();lp=z-torch.logsumexp(z,-1);v,k=lp.topk(5);target=r['token_ids'][x['index']]
    assert torch.isfinite(lp).all()
    reports.append(dict(key=x['key'],target_id=target,target_logp=float(lp[target]),top5_ids=k.tolist(),top5_logp=v.tolist()))
 result=dict(engine=a.engine,rows=reports,seconds=time.monotonic()-start,plan_sha256=sha(a.input/'plan.json'),steering=False,lexical_penalty=False,optimizer_steps=0)
 (a.output/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except BaseException:
 import traceback
 (a.output/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),seconds=time.monotonic()-start)));raise
