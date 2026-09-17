"""Descriptive frozen-calibration audit. No gate fitting or generation."""
from pathlib import Path
import json,hashlib,tarfile,math,re,sys
import numpy as np
ROOT=Path('E:/srtp/hybrid-syncthink-cgrs-20260912');B=ROOT/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(B))
from calibrate_penalty_scale_cpu import vocabulary
from audit_strength_coupling import coefficients
W=Path('E:/srtp/srtp-final/.codex_work');fit=W/'auto_code_v2_500_20260908/fit.json';archive=W/'own_calibration_500_20260908/calibration_assets.tar.gz';tok=W/'label_audit_30_20260910/tokenizer.json';native=ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
f=json.loads(fit.read_text());pieces,boundaries=vocabulary(tok)
with tarfile.open(archive) as t:raw=t.extractfile('generations.jsonl').read()
assert hashlib.sha256(raw).hexdigest()==f['calibration_source_sha256'];rows=[json.loads(s) for s in raw.splitlines()];steps=[]
for q,row in enumerate(rows):
 previous=None;probs=[];start=0
 for pos,(tid,logp) in enumerate(zip(row['token_ids'],row['logprobs'])):
  if tid==151649:break
  if tid not in boundaries:probs.append(math.exp(logp));continue
  if probs:
   mean=float(np.mean(probs));text=''.join(pieces[i] for i in row['token_ids'][start:pos]);following=''.join(pieces[i] for i in row['token_ids'][pos+1:pos+100]);clean=not pieces[tid].rsplit('\n\n',1)[-1].strip()
   minimum_piece=pieces[row['token_ids'][start+int(np.argmin(probs))]]
   category='word' if re.fullmatch(r'[A-Za-z]+',minimum_piece.strip()) else 'number' if re.fullmatch(r'[0-9]+',minimum_piece.strip()) else 'punctuation_or_mixed'
   steps.append(dict(question=q,train_index=row['train_index'],start=start,end=pos,n=len(probs),mean=mean,between=0 if previous is None else (mean-previous)**2/4,within=float(np.var(probs)),minimum=min(probs),minimum_piece=minimum_piece,minimum_category=category,clean=clean,next_reflection=bool(re.match(r'\s*(?:Wait|wait|But|but|Alternatively|Alternative|Hmm)\b',following)),text=text,following=following))
   previous=mean
  probs=[];start=pos+1
c=np.array([s['mean'] for s in steps]);v=np.array([s['between'] for s in steps]);within=np.array([s['within'] for s in steps]);n=np.array([s['n'] for s in steps]);q=np.array([s['question'] for s in steps]);coefs=coefficients(c,v,f['parameters'],native);eligible=(coefs<0)&np.array([s['clean'] for s in steps]);threshold=float(np.quantile(within,.75));high=within>=threshold
for s,a in zip(steps,coefs):s['analytical_coefficient']=float(a)
# Fixed, descriptive question-grouped linear projection; no generation-policy model.
L=np.log1p(n);X=np.column_stack([np.ones(len(c)),c,c*c,L,L*L,c*L,v]);counts=np.bincount(q);weight=1/counts[q];pred=np.empty(len(c));models=[]
for fold in range(5):
 test=q%5==fold;train=~test;sw=np.sqrt(weight[train]);beta=np.linalg.lstsq(X[train]*sw[:,None],within[train]*sw,rcond=None)[0];pred[test]=X[test]@beta;models.append(beta.tolist())
mu=np.average(within,weights=weight);r2=1-np.sum(weight*(within-pred)**2)/np.sum(weight*(within-mu)**2)
summary=dict(questions=500,steps=len(steps),clean_negative_step_boundaries=int(eligible.sum()),descriptive_variance_q75=threshold,eligible_above_variance_q75=int((eligible&high).sum()),eligible_high_variance_questions=len(set(q[eligible&high].tolist())),mean_within_variance_correlation=float(np.corrcoef(c,within)[0,1]),mean_length_between_projection_question_weighted_out_of_fold_r2=float(r2),projection_basis=['1','mean','mean_squared','log1p_length','log1p_length_squared','mean_x_log1p_length','between_step_variance'],projection_folds='fixed question ordinal modulo5; analytical predictability only, not correctness validation',minimum_token_categories={k:sum(s['minimum_category']==k for s,e,h in zip(steps,eligible,high) if e and h) for k in ['word','number','punctuation_or_mixed']},limitations=['No reasoning-correctness labels at steps. Next reflection occurrence is not necessary reflection.','Greedy saved trajectories under no intervention, analytical float64 controller. Not live RC14 or counterfactual gate replay.','Variance q75 is a descriptive stratum; NOT a chosen intervention threshold.','Mean constrains variance to at most mean*(1-mean). Explained variance is not gate efficacy.','No GPU, no new answers, no test-set threshold selection.'])
cases=[]
for reflection in [True,False]:
 pool=[s for s,e,h in zip(steps,eligible,high) if e and h and s['next_reflection']==reflection]
 pool.sort(key=lambda s:hashlib.sha256(f"dispersion-audit-v1|{s['train_index']}|{s['end']}".encode()).hexdigest());chosen=set()
 for s in pool:
  if s['question'] in chosen:continue
  chosen.add(s['question']);cases.append(dict(s,problem=rows[s['question']]['problem']))
  if len(chosen)==4:break
out=B/'dispersion_feasibility_20260917';out.mkdir(exist_ok=False)
summary['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fit,archive,tok,native]};summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();summary['manual_case_selection']='4 next-reflection and4 non-reflection examples, hash order, distinct questions within each stratum; not representative prevalence'
(out/'audit.json').write_text(json.dumps(summary,indent=2));(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));(out/'steps.json').write_text(json.dumps([{k:v for k,v in s.items() if k not in ['text','following']} for s in steps]));print(json.dumps(summary,indent=2));print('CASES')
for s in cases:print(json.dumps({k:s[k] for k in ['train_index','end','n','mean','within','minimum_piece','next_reflection','text','following']},ensure_ascii=False))
