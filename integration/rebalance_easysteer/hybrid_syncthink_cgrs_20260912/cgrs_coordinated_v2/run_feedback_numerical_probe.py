"""Targeted inference-only score discrepancy diagnostic; never optimize/generate."""
import argparse,hashlib,json,math,signal,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s diagnostic ceiling')));signal.alarm(600)
 try:
  plan=json.loads((a.input/'plan.json').read_text());original=Path(plan['original_input']);base=json.loads((original/'plan.json').read_text())
  assert sha(original/'plan.json')==plan['original_plan_sha256']
  for n,h in base['input_sha256'].items():assert sha(original/n)==h,n
  for n,h in plan['input_sha256'].items():assert sha(a.input/n)==h,n
  assets=base['assets']
  for n,v in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==v['sha256']
  assert sha(Path(assets['vector']['path']))==assets['vector']['sha256']
  import numpy as np,torch,sys
  sys.path.insert(0,str(original))
  from transformers import AutoModelForCausalLM,AutoTokenizer
  from feedback_fixed_history import FixedHistoryScorer,effective_scales
  model=AutoModelForCausalLM.from_pretrained(assets['model_path'],torch_dtype=getattr(torch,plan.get('hf_dtype','bfloat16')),attn_implementation=plan.get('hf_attention','sdpa'),local_files_only=True).cuda().eval()
  for p in model.parameters():p.requires_grad_(False)
  tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
  boundaries=torch.tensor(sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s),device='cuda')
  vec=torch.load(assets['vector']['path'],weights_only=True,map_location='cuda').float()
  tables={k:torch.as_tensor(v,device='cuda') for k,v in dict(np.load(original/'opening.npz')).items()};tables['candidate_ids']=tables['candidate_ids'].long()
  scorer=FixedHistoryScorer(model,vec,boundaries,tables)
  rows={r['key']:r for r in json.loads((original/'rows.json').read_text())};reports=[]
  with torch.inference_mode():
   for item in plan['targets']:
    row=rows[item['key']];i=item['index'];p=len(row['prompt_token_ids'])
    trace_path=Path(base['replay_directory'])/('scoring_capture_'+row['key']+'.npz');assert sha(trace_path)==item['trace_sha256']
    trace=dict(np.load(trace_path));ids=torch.tensor(row['prompt_token_ids']+row['token_ids'][:i],device='cuda')
    scales=effective_scales(ids,torch.as_tensor(trace['history'][:len(ids)],device='cuda'),p,boundaries)
    target=row['token_ids'][i];results={}
    for mode in plan.get('modes',['full','full_repeat','cached_128','cached_1']):
     torch.cuda.synchronize();t=time.monotonic();past=None
     if mode.startswith('full'):
      scorer.scales=scales;h=model.model(input_ids=ids[None],use_cache=False,return_dict=True).last_hidden_state[0,-1]
     else:
      chunk=int(mode.split('_')[1]);pos=0
      while pos<len(ids):
       end=min(len(ids),p if pos==0 else pos+chunk);scorer.scales=scales[pos:end]
       o=model.model(input_ids=ids[pos:end][None],past_key_values=past,use_cache=True,return_dict=True)
       past=o.past_key_values;h=o.last_hidden_state[0,-1];pos=end
     z=model.lm_head(h).float();raw=float(torch.softmax(z,-1).max());c=tables['candidate_ids'];state=int(trace['lex_state'][i]);gate=bool(trace['lex_gate'][i]);hits=tables['hits'][state,tables['token_classes'][c].long()] & gate
     z[c]=torch.where(hits,z[c]-math.log(2),z[c]);lp=z-torch.logsumexp(z,-1);v,k=torch.topk(lp,5)
     torch.cuda.synchronize();results[mode]=dict(target_logp=float(lp[target]),raw_max_probability=raw,top5_ids=k.cpu().tolist(),top5_logp=v.cpu().tolist(),seconds=time.monotonic()-t)
     del past
    reports.append(dict(key=row['key'],index=i,prefix_input_tokens=len(ids),target_id=target,context=tok.decode(row['token_ids'][max(0,i-20):i+20]),native_logp=float(trace['logp'][i]),native_rawmax=float(trace['rawmax'][i]),saved_full_hf_logp=item['saved_full_hf_logp'],results=results))
    (a.output/'progress.json').write_text(json.dumps(reports,indent=2)+'\n');print(row['key'],results,flush=True)
  scorer.close();(a.output/'complete.json').write_text(json.dumps(dict(status='Inference-only prefix diagnostic; no new answers/gradients/update',rows=reports,wall_seconds=time.monotonic()-start,plan_sha256=sha(a.input/'plan.json')),indent=2)+'\n')
 except BaseException:
  (a.output/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),wall_seconds=time.monotonic()-start),indent=2));raise
if __name__=='__main__':main()
