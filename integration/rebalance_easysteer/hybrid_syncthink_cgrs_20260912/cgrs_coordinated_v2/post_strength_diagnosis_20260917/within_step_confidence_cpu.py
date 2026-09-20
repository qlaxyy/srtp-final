from pathlib import Path
import sys,json,tarfile,math,hashlib
import numpy as np
root=Path('E:/srtp/hybrid-syncthink-cgrs-20260912');b=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(b));from calibrate_penalty_scale_cpu import vocabulary
w=Path('E:/srtp/srtp-final/.codex_work');tok=w/'label_audit_30_20260910/tokenizer.json';fit=w/'auto_code_v2_500_20260908/fit.json';archive=w/'own_calibration_500_20260908/calibration_assets.tar.gz';f=json.loads(fit.read_text());pieces,boundaries=vocabulary(tok)
with tarfile.open(archive) as t:raw=t.extractfile('generations.jsonl').read()
assert hashlib.sha256(raw).hexdigest()==f['calibration_source_sha256'];steps=[]
for q,line in enumerate(raw.splitlines()):
 row=json.loads(line);probs=[]
 for pos,(token,logp) in enumerate(zip(row['token_ids'],row['logprobs'])):
  if token==151649:break
  if token not in boundaries:probs.append(math.exp(logp));continue
  if probs:
   steps.append(dict(question=q,end=pos,n=len(probs),mean=sum(probs)/len(probs),minimum=min(probs),within_variance=float(np.var(probs)),below_half=sum(x<.5 for x in probs)));probs=[]
means=np.array([s['mean'] for s in steps]);high=[s for s in steps if s['mean']>=f['parameters']['q75c']];mixed=[s for s in steps if s['mean']>=f['parameters']['q25c'] and s['below_half']>0]
d=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [tok,fit,archive]},questions=500,steps=len(steps),high_confidence_threshold_frozen=f['parameters']['q75c'],high_confidence_steps=len(high),high_steps_containing_token_probability_below_half=sum(s['below_half']>0 for s in high),above_q25_with_below_half=dict(steps=len(mixed),questions=len({s['question'] for s in mixed}),fraction_of_steps=len(mixed)/len(steps)),within_step_variance_quantiles=np.quantile([s['within_variance'] for s in steps],[0,.25,.5,.75,1]).tolist(),limitations=['Original frozen greedy calibration, not test output and not sampled deployment distribution.','Float64 descriptive calculations; not exact native GPU replay.','Below0.5 is descriptive token uncertainty, not wrong-reasoning labels or a chosen runtime gate.','Within-step variance partly depends on syntax, length and tokenization; no evidence yet of error detection or beneficial compression.'],gpu_calls=0)
out=b/'post_strength_diagnosis_20260917';(out/'within_step_confidence_audit.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
