"""Frozen calibration audit: balanced boxes, exact byte/token boundaries."""
import bisect, hashlib, json, re, tarfile
from pathlib import Path
from prepare_length_vector import HERE, ROOT, read, save, sha
from review_wsc_native_cpu import Decoder

def boxes(raw):
    for m in re.finditer(rb'\\boxed\s*\{', raw):
        depth=1; pos=m.end()
        while pos<len(raw) and depth:
            if raw[pos]==123: depth+=1
            elif raw[pos]==125: depth-=1
            pos+=1
        if not depth:
            yield m.start(),pos,raw[m.end():pos-1].decode('utf8')

def main():
    assert list(boxes(b'x \\boxed{\\frac{1}{2}} y'))==[(2,21,'\\frac{1}{2}')]
    out=HERE/'answer_anchor_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');byq=[[] for _ in range(500)]
    for i,s in enumerate(steps):byq[s['question']].append((i,s))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tf:raw=tf.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()==audit['source_sha256']
    rows=[json.loads(s) for s in raw.splitlines()]
    trainpath=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    assert sha(trainpath)=='4bd7f5266f6c5c4fc729da2d9cbefbe2453cf8df7bd6ba93dc02b9e11512cefd'
    train=[json.loads(s) for s in trainpath.read_text(encoding='utf8').splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');records=[]
    for q,r in enumerate(rows):
        ids=r['token_ids'];n=ids.index(151649) if 151649 in ids else len(ids)
        chunks=[dec.added[t].encode() if t in dec.added else bytes(dec.bytes[c] for c in dec.vocab[t]) for t in ids[:n]]
        ends=[0]
        for chunk in chunks:ends.append(ends[-1]+len(chunk))
        rawtext=b''.join(chunks);assert rawtext.decode('utf8',errors='replace')==dec.decode(ids[:n])
        bs=[dict(byte_start=a,byte_stop=b,token_stop=bisect.bisect_left(ends,b),answer=v) for a,b,v in boxes(rawtext)]
        seen={};repeats=[]
        for i,s in byq[q]:
            if s['stop']>n or s['stop']-s['start']<16:continue
            text=' '.join(dec.decode(ids[s['start']:s['stop']]).lower().split())
            if text in seen:repeats.append(dict(step=i,first_step=seen[text],start=s['start'],stop=s['stop'],confidence=s['confidence']))
            else:seen[text]=i
        records.append(dict(question=q,train_index=r['train_index'],problem_sha256=audit['questions'][q]['problem_sha256'],
            thinking_tokens=n,boxes=bs,repeats=repeats,gold_row=train[r['train_index']]))
    refs=read(HERE/'length_sign_ablation_20260918_run1/historical_compact.json')
    save(out/'input.json',dict(source_sha256=audit['source_sha256'],steps_sha256=sha(cache/'steps.json'),
        script_sha256=sha(Path(__file__)),grader_sha256=refs['grader_sha256'],rows=records))
    summary=dict(questions=500,questions_with_box=sum(bool(r['boxes']) for r in records),boxes=sum(len(r['boxes']) for r in records),
        questions_with_repeat_after_box=sum(bool(r['boxes']) and any(s['start']>=r['boxes'][0]['token_stop'] for s in r['repeats']) for r in records),
        input_sha256=sha(out/'input.json'),new_forward=0,new_generation=0)
    save(out/'coverage.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
