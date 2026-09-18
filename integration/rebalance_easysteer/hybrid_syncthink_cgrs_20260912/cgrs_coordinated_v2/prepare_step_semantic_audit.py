"""Create a fixed source-stratified CPU audit sample from exact saved tokens."""
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    home=Path(__file__).resolve().parent
    root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    out=home/'paired_step_semantic_audit_20260918_run1';out.mkdir(exist_ok=False)
    p=root/'.codex_work/wsc_weights_20260917/tokenizer.json';b=p.read_bytes()
    assert hashlib.sha256(b).hexdigest()=='88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773'
    tok=json.loads(b);vocab={v:k for k,v in tok['model']['vocab'].items()}
    # Invert the byte-level alphabet from the tokenizer's ByteLevel decoder.
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for x in range(256):
        if x not in bs:bs.append(x);cs.append(256+n);n+=1
    inv={chr(c):b for b,c in zip(bs,cs)}
    def decode(ids):
        return bytes(inv[c] for i in ids for c in vocab[i]).decode('utf8',errors='replace')
    rows=json.loads((home/'outcome_endpoints_20260918_run1/rows.json').read_text())
    rows={(r['question'],r['side']):r for r in rows}
    tf=tarfile.open(root/'.codex_work/outcome_endpoints_20260918_evidence.tar.gz')
    m=json.load(tf.extractfile('outcome_endpoints_20260918_run1/features/manifest.json'))
    m={(r['question'],r['side']):r for r in m}
    base=json.loads((home/'self_feedback_train500_20260918_run1/baseline.json').read_text())
    tt=tarfile.open(root/'.codex_work/self_feedback_train500_20260918_evidence.tar.gz')
    l27=json.load(tt.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    sources={'U':{r['dataset_index']:r for r in base},'L27':{r['dataset_index']:r for r in l27}}
    for (q,side),r in rows.items():
        text=decode(r['token_ids']);original=sources[r['source']][q]['text']
        assert original.startswith(text),(q,side,'exact decode mismatch')
    selections=json.loads((home/'paired_endpoint_diagnostic_20260918_run1/question_selection.json').read_text())['post_divergence_matched']
    chosen=[]
    for source in ['U','L27']:
        chosen+=sorted([r for r in selections if r['long_source']==source],key=lambda r:r['problem_sha256'])[:6]
    records=[]
    for r in chosen:
        q=r['question']; pairs=list(zip(r['long_step_indices'],r['short_step_indices']))
        # Fixed median of the long-step index order, independent of text content.
        i,j=sorted(pairs)[len(pairs)//2]
        record=dict(question=q,problem_sha256=r['problem_sha256'],long_source=r['long_source'],
            problem=sources['U'][q]['problem'],selection='median matched pair by long step index')
        for side,index in [('long',i),('short',j)]:
            rr=rows[q,side];steps=m[q,side]['steps'];s=steps[index]
            context=[]
            for k in range(max(0,index-1),min(len(steps),index+2)):
                z=steps[k];context.append(dict(step=k,target=k==index,text=decode(rr['token_ids'][z['start']:z['stop']])))
            record[side]=dict(source=rr['source'],step=index,confidence=s['confidence'],lexical_hit=s['lexical_hit'],
                total_thinking_tokens=len(rr['token_ids']),context=context,
                full_thinking_text=decode(rr['token_ids']))
        records.append(record)
    protocol=dict(status='fixed sample before semantic review',count=12,strata='6 long-U and 6 long-L27; smallest normalized problem SHA256 within source',
        pair_selection='median matched pair sorted by long step index',decoding='ByteLevel inverse; all478 full saved thinking prefixes exactly verified against original answer text',
        labels=['same_subproblem','same_stage_different_subproblem','different_stage','unclear'],
        redundancy='Only record plausible redundant repetition with explicit content evidence; no causal deletion claim.',
        limits='Small source-balanced diagnostic sample, not population prevalence or a trained semantic judge.',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,obj in [('protocol',protocol),('sample',records)]:
        (out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Exact-decoded 478 saved thinking prefixes; fixed 12-question audit sample.')
    for r in records:
        print('\nQUESTION',r['question'],r['long_source'],r['problem'])
        for side in ['long','short']:
            print(side,r[side]['step'],r[side]['confidence'])
            for s in r[side]['context']:print('TARGET' if s['target'] else 'CONTEXT',s['text'])


if __name__=='__main__':main()
