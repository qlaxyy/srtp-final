"""Fixed offline loop hypothesis. No edits to frozen ReBalance or RC14."""
import json
from pathlib import Path
import numpy as np
from engineering import HERE,save,sha

def first_loop(tokens):
    # These bounds are design choices fixed before the audit, not tuned on outcomes.
    # Three adjacent identical blocks, length8..64, >=4 distinct token IDs.
    n=len(tokens);prefix=[0];h=0;mask=(1<<64)-1;base=1000003
    for x in tokens:h=(h*base+int(x)+1)&mask;prefix.append(h)
    prefix=np.array(prefix,dtype=np.uint64);best=None
    for period in range(8,min(64,n//3)+1):
        power=np.uint64(pow(base,period,1<<64))
        hashes=prefix[period:]-prefix[:-period]*power
        hits=np.flatnonzero((hashes[2*period:]==hashes[period:-period]) & (hashes[period:-period]==hashes[:-2*period]))
        for start in hits:
            start=int(start);end=start+3*period
            if best is not None and end>=best['end']:break
            a=tokens[start:start+period]
            if len(set(a))<4 or a!=tokens[start+period:start+2*period] or a!=tokens[start+2*period:end]:continue
            continued=0
            while end+continued<n and tokens[end+continued]==a[continued%period]:continued+=1
            best=dict(start=start,end=end,period=period,block_token_ids=a,subsequent_exact_loop_tokens=continued,remaining_thinking_tokens=n-end)
            break
    return best

def main():
    raw=Path('E:/srtp/B-repeatability-stopped_0916');out=HERE/'reasoning_tradeoffs_cpu_20260916'
    manifest=json.loads((raw/'manifest.json').read_text());groups={};cases=[];inputs={}
    for seed in (142,242):
        for arm in ('R','RC14','RChistory'):
            folder=raw/'results'/f's{seed}_{arm}';p=folder/'result.json';g=folder/'stopped_batch_author_labels.jsonl'
            for path in (p,g):assert sha(path)==manifest[path.relative_to(raw).as_posix()];inputs[str(path)]=sha(path)
            result=json.loads(p.read_text());labels={r['dataset_index']:r for r in map(json.loads,g.read_text().splitlines())};hits=[]
            for r in result['records']:
                hit=first_loop(r['token_ids'][:r['thinking_tokens']])
                if hit:
                    item=dict(seed=seed,arm=arm,train_index=r['train_index'],problem_sha256=r['problem_sha256'],correct=labels[r['dataset_index']]['correct'],capped=r['finish_reason']=='length',thinking_tokens=r['thinking_tokens'],**hit)
                    hits.append(item);cases.append(dict(**item,problem=r['problem'],text=r['text']))
            groups[f's{seed}_{arm}']=dict(n=100,covered=len(hits),finally_wrong=sum(not r['correct'] for r in hits),capped=sum(r['capped'] for r in hits),exact_continuation_tokens=sum(r['subsequent_exact_loop_tokens'] for r in hits),rows=hits)
    save(out/'contiguous_loop_cases.json',cases)
    save(out/'contiguous_loop_audit.json',dict(rule='Three immediately adjacent identical token blocks, block length8..64, at least4 distinct tokens, thinking only; first match only. No threshold scan or label input to detector.',source_sha256=sha(Path(__file__),True),input_sha256=inputs,groups=groups,limitations=['A trigger is not a correct stopping point or a proven opportunity to save the remaining tokens.','Repetition can be valid computation; changing next-token distribution can damage reasoning.','600 trajectories from100 exposed questions; no independent validation.','No per-step R coefficient trace here, so eligibility of any extra R-dependent condition is unknown.'],gpu_ready=False))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in groups.items()},indent=2))

if __name__=='__main__':main()
