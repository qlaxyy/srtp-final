"""Conservative CPU retrieval of repeated spans, requiring subsequent semantic review."""
import difflib
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path


def main():
    home=Path(__file__).resolve().parent
    root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    out=home/'repeated_derivation_20260918_run1';out.mkdir(exist_ok=False)
    protocol=dict(status='retrieval only, not step labels or a deployable vector',
        parents='All231 existing CC length-gap-selected training pairs',
        retrieval='Long-side blank-line spans >=32 tokens; nonadjacent step pair; shared consecutive block >=24 tokens; total exact matched tokens >=50% of shorter span; no confidence filter',
        reference='Highest exact-token overlap short-side span >=32 tokens, retrieval only',
        sample='Up to6 parents per long-source stratum ordered by problem SHA; highest overlap candidate per question',
        exclusions='No automatic classification of necessary correction, repeated numbers or answer mentions as redundant.',
        data='Reuse saved tokens and states; no generation, GPU, test-set fit or function change.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    tokpath=root/'.codex_work/wsc_weights_20260917/tokenizer.json'
    assert hashlib.sha256(tokpath.read_bytes()).hexdigest()=='88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773'
    tok=json.loads(tokpath.read_bytes()); vocab={v:k for k,v in tok['model']['vocab'].items()}
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for i in range(256):
        if i not in bs:bs.append(i);cs.append(256+n);n+=1
    inv={chr(c):b for b,c in zip(bs,cs)}
    def decode(ids):return bytes(inv[c] for i in ids for c in vocab[i]).decode('utf8',errors='replace')
    old=home/'outcome_endpoints_20260918_run1'
    rows={(r['question'],r['side']):r for r in json.loads((old/'rows.json').read_text())}
    archive=root/'.codex_work/outcome_endpoints_20260918_evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='ba391343b8b86e28514f4dc638a7bc189979883727a17b3fe0f1b757bfce22ff'
    t=tarfile.open(archive);m=json.load(t.extractfile('outcome_endpoints_20260918_run1/features/manifest.json'))
    manifests={(r['question'],r['side']):r for r in m}
    candidates=[]
    for q in sorted({r['question'] for r in m if r['kind']=='CC'}):
        lr=rows[q,'long'];sr=rows[q,'short'];lm=manifests[q,'long'];sm=manifests[q,'short']
        def spans(r,m):return {i:r['token_ids'][s['start']:s['stop']] for i,s in enumerate(m['steps']) if s['stop']-s['start']>=32}
        long=spans(lr,lm);short=spans(sr,sm);index=defaultdict(set);pairs=set()
        for i,ids in long.items():
            grams={tuple(ids[k:k+24]) for k in range(len(ids)-23)}
            for g in grams:
                for j in index[g]:
                    if i-j>=2:pairs.add((j,i))
                index[g].add(i)
        for first,repeat in sorted(pairs):
            aa,bb=long[first],long[repeat];blocks=difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_matching_blocks()
            total=sum(b.size for b in blocks);coverage=total/min(len(aa),len(bb))
            if max(b.size for b in blocks)<24 or coverage<.5:continue
            refs=[]
            for j,ids in short.items():
                b=difflib.SequenceMatcher(None,aa,ids,autojunk=False).get_matching_blocks()
                refs.append((sum(z.size for z in b)/max(len(aa),len(ids)),j))
            score,j=max(refs) if refs else (0,None)
            c=dict(question=q,problem_sha256=lr['problem_sha256'],long_source=lr['source'],first_step=first,repeat_step=repeat,
                matched_tokens=total,coverage=coverage,first_tokens=len(aa),repeat_tokens=len(bb),short_reference_step=j,short_reference_overlap=score)
            candidates.append(c)
    sample=[]
    for source in ['U','L27']:
        qs=sorted({c['question'] for c in candidates if c['long_source']==source},key=lambda q:rows[q,'long']['problem_sha256'])[:6]
        for q in qs:
            c=max([c for c in candidates if c['question']==q],key=lambda c:(c['coverage'],c['matched_tokens'],-c['repeat_step']))
            r=dict(c)
            for key,side,ix in [('first','long',c['first_step']),('repeat','long',c['repeat_step']),('short_reference','short',c['short_reference_step'])]:
                rr=rows[q,side];mm=manifests[q,side];r[key]=[]
                if ix is not None:
                    for k in range(max(0,ix-1),min(len(mm['steps']),ix+2)):
                        s=mm['steps'][k];r[key].append(dict(step=k,target=k==ix,text=decode(rr['token_ids'][s['start']:s['stop']])))
            sample.append(r)
    summary=dict(status='CPU retrieval complete; all candidates unlabelled until review',candidate_pairs=len(candidates),parents=len({c['question'] for c in candidates}),
        source_parent_counts={s:len({c['question'] for c in candidates if c['long_source']==s}) for s in ['U','L27']},sample_questions=[s['question'] for s in sample],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,obj in [('candidates',candidates),('sample',sample),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary))
    for r in sample:
        print('\nQUESTION',r['question'],r['long_source'])
        for key in ['first','repeat','short_reference']:
            print(key)
            for s in r[key]:print('TARGET' if s['target'] else 'CONTEXT',s['step'],s['text'])


if __name__=='__main__':main()
