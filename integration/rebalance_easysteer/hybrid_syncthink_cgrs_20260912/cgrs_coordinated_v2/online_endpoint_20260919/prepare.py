"""Descriptive endpoint screening; no GPU or efficacy-based threshold selection."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
HOME=Path(__file__).resolve().parent
HERE=HOME.parent
ROOT=HERE.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    run=HERE/'iterative_recalibration_20260919_run1'
    labels={'R':read(run/'R_training_author_analysis.json')['records'],
      'L27':[r['L27'] for r in read(HERE/'self_feedback_train500_20260918_run1/labels.json')]}
    result={}
    archive=ROOT/'.codex_work/iterative_recalibration_20260919/calibration_evidence.tar.gz'
    with tarfile.open(archive) as t:
      for name,lab in labels.items():
        s=json.load(t.extractfile(name+'_fit/steps.json'))
        p=read(run/(name+'_fit')/'fit.json')['parameters']
        c=np.array([x['confidence'] for x in s]);v=np.array([x['variance'] for x in s]);q=np.array([x['question'] for x in s]);lex=np.array([x['lexical_hit'] for x in s])
        base=(~lex)&(c>p['q75c']);closed=np.array([lab[i]['tokens']<16000 for i in q])
        masks={'original':base,'c90':(~lex)&(c>np.quantile(c,.90)),
          'c95':(~lex)&(c>np.quantile(c,.95)),
          'low_variance':base&(v<p['q25v']),
          'exclude_capped':base&closed,
          'exclude_capped_low_variance':base&closed&(v<p['q25v'])}
        result[name]={}
        for k,mask in masks.items():
          selected=q[mask];counts=np.bincount(selected,minlength=500)
          result[name][k]=dict(steps=int(mask.sum()),questions=int((counts>0).sum()),
            top10_steps=int(np.sort(counts)[-10:].sum()),
            from_capped=int(sum(lab[i]['tokens']==16000 for i in selected)),
            from_wrong=int(sum(not lab[i]['correct'] for i in selected)),
            effective_questions=float(counts.sum()**2/(counts@counts)) if counts.sum() else 0)
    save(HOME/'screen.json',dict(results=result,source_sha256=sha(archive),
      limitation='BASE replay only. Correctness is descriptive, never a selection input. No new vector or efficacy claim.'))
    rows=read(HERE/'iterative_recalibration_20260919/rows.json')[:8]
    save(HOME/'rows.json',rows)
    assets=read(HERE/'iterative_recalibration_20260919/assets.json')
    save(HOME/'plan.json',dict(phase='engineering_only',model=assets['model'],assets=assets,
      arms=['R_off','R_observe','L27_off','L27_observe'],rows=8,max_tokens=512,seed=42,
      temperature=0,top_p=1,process_limit_seconds=900,max_answers=32,max_generated_tokens=16384,
      runtime_hashes=read(HERE/'iterative_recalibration_20260919/runtime_hashes.json'),
      source_rows_sha256=sha(HOME/'rows.json'),
      no_fitting=True,no_full_generation=True,
      gate='Exact per-arm on/off tokens, pre-penalty native pmax and controller history; exact positions; finite pre/post hidden; nonzero steering observed.',
      limitation='Eager synchronous engineering only; not throughput or compression evidence. No automatic expansion.'))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
