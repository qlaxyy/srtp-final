"""CPU-only grading and endpoint diagnosis after all 1000 traces complete."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def spans(ids,bounds):
    closed=151649 in ids;n=ids.index(151649) if closed else len(ids);first=0;out=[]
    for stop in range(n+1):
        if not (stop<n and ids[stop] in bounds) and not (stop==n and closed):continue
        if stop>first:out.append((first,stop))
        first=stop+1
    return out,n
def composition(mask,questions,labels):
    q=questions[mask];counts=np.bincount(q,minlength=500)
    return dict(steps=len(q),questions=int((counts>0).sum()),from_capped=sum(int(labels[i]['capped']) for i in q),
      from_wrong=sum(int(not labels[i]['correct']) for i in q),top10_steps=int(np.sort(counts)[-10:].sum()),
      effective_questions=float(counts.sum()**2/(counts@counts)) if len(q) else 0)
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--recovery',type=Path)
    p.add_argument('--author-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();home=Path(__file__).resolve().parent
    if a.recovery:
        assert (a.run/'failure.json').exists() and not (a.run/'complete.json').exists()
        assert read(a.recovery/'complete.json')['passed']
        assert not (a.run/'L27_batch448/result.json').exists()
    else:assert read(a.run/'complete.json')['passed']
    rows=read(home/'rows.json');assert len(rows)==500
    out=a.output;out.mkdir(exist_ok=False);began=time.monotonic()
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import extract_answer,parse_ground_truth
    from utils.grader import check_is_correct
    from transformers import AutoTokenizer
    plan=read(home/'plan.json');tok=AutoTokenizer.from_pretrained(plan['assets']['model_path'],local_files_only=True)
    bounds={i for s,i in tok.get_vocab().items() if 'ĊĊ' in s}
    spec=importlib.util.spec_from_file_location('author_labels',a.author_source)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    all_results={};file_hashes={}
    for source in ['R','L27']:
        labels=[];steps=[];features={'pre':[],'post':[]};total_seconds=0;rounding=[]
        for offset in range(0,500,64):
            root=a.recovery if a.recovery and source=='L27' and offset==448 else a.run
            folder=root/(source+'_batch'+str(offset));d=read(folder/'result.json');total_seconds+=d['generation_seconds']
            assert len(d['records'])==min(64,500-offset)
            for j,r in enumerate(d['records']):
                q=offset+j;expected=rows[q];assert r['problem_sha256']==expected['problem_sha256']
                ids=r['token_ids'];native_path=folder/f'{q}_native.npy';hidden_path=folder/f'{q}_hidden.npz'
                n=np.load(native_path);z=np.load(hidden_path);assert len(n)==len(ids)<=16000
                assert n[:,0].astype(int).tolist()==ids and hashlib.sha256(n.tobytes()).hexdigest()==r['native_sha256']
                assert np.isfinite(n[:,1]).all()
                assert len(set(z['starts'].tolist()))==len(z['starts'])
                starts={int(s):i for i,s in enumerate(z['starts'])}
                states={mode:z[mode+'_hidden'] for mode in ['pre','post']}
                segments,thinking=spans(ids,bounds)
                _,gold=parse_ground_truth(expected,'math');answer=extract_answer(tok.decode(ids,skip_special_tokens=True))
                lab=dict(question=q,problem_sha256=r['problem_sha256'],correct=bool(check_is_correct(answer,gold)),
                  tokens=len(ids),thinking_tokens=thinking,capped=len(ids)==16000)
                labels.append(lab);prev=None
                for start,stop in segments:
                    assert start in starts,('Missing actual step-first state',source,q,start)
                    k=starts[start];c=float(n[start:stop,1].mean())
                    steps.append(dict(question=q,start=start,stop=stop,confidence=c,variance=0 if prev is None else (c-prev)**2/4,
                      lexical_hit=bool(author.has_lexicon_hit(tok.decode(ids[start:stop])))))
                    for mode in ['pre','post']:features[mode].append(states[mode][k])
                    if stop<len(ids) and ids[stop] in bounds:rounding.append(abs(c-float(n[stop,3])))
                    prev=c
                for path in [native_path,hidden_path]:file_hashes[str(path)]=sha(path)
        assert len(labels)==500
        c=np.array([s['confidence'] for s in steps]);v=np.array([s['variance'] for s in steps]);q=np.array([s['question'] for s in steps]);lex=np.array([s['lexical_hit'] for s in steps])
        q25,q75=np.quantile(c,[.25,.75]);O=lex|(c<q25);U=(~lex)&(c>q75);strict=U&np.array([not labels[i]['capped'] for i in q])
        diag={};vectors={}
        for mode,vals in features.items():
            x=np.stack(vals).astype(np.float64);assert np.isfinite(x).all()
            for name,mask in [('original',U),('exclude_capped',strict)]:
                key=mode+'_'+name
                if not mask.any() or not O.any():diag[key]={'insufficient':True};continue
                direction=x[O].mean(0)-x[mask].mean(0);vectors[key]=direction
                diag[key]=dict(vector_norm=float(np.linalg.norm(direction)),over_norm=float(np.linalg.norm(x[O].mean(0))),under_norm=float(np.linalg.norm(x[mask].mean(0))))
        result=dict(summary=dict(correct=sum(x['correct'] for x in labels),mean_tokens=float(np.mean([x['tokens'] for x in labels])),
          mean_thinking_tokens=float(np.mean([x['thinking_tokens'] for x in labels])),capped=sum(x['capped'] for x in labels),generation_seconds=total_seconds),
          steps=len(steps),confidence_quantiles=[q25,q75],variance_quantiles=np.quantile(v,[.25,.75]).tolist(),
          over=composition(O,q,labels),under_original=composition(U,q,labels),under_exclude_capped=composition(strict,q,labels),
          endpoints=diag,direction_cosines={a+'__'+b:cosine(vectors[a],vectors[b]) for a,b in [('pre_original','pre_exclude_capped'),('pre_original','post_original'),('pre_exclude_capped','post_exclude_capped')] if a in vectors and b in vectors},
          max_native_step_mean_rounding_error=max(rounding) if rounding else None,
          limitation='Training descriptive only. No new vector deployed; no dynamic fitting; different native trajectories cannot be paired with historical base replay states.')
        save(out/(source+'_labels.json'),labels);save(out/(source+'_steps.json'),steps);save(out/(source+'_diagnosis.json'),result);all_results[source]=result
    save(out/'summary.json',dict(groups=all_results,cpu_seconds=time.monotonic()-began,
      collection_status='Recovered training collection; not one continuous successful run' if a.recovery else 'One complete training collection',
      main_run=str(a.run),recovery_run=str(a.recovery) if a.recovery else None,author_source_sha256=sha(a.author_source)));save(out/'source_hashes.json',file_hashes)
    print(json.dumps(all_results,indent=2))
if __name__=='__main__':main()
