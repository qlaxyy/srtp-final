"""Deterministic cross-question qualitative audit, not semantic gold."""
import hashlib,json,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'repeat_content_review_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    src=HERE/'answer_anchor_20260918_cpu/input.json';data=read(src)
    steps=read(ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json')
    groups={name:[r for r in data['rows'] if r['repeats'] and (r['thinking_tokens']==16000)==cap] for name,cap in [('capped',True),('not_capped',False)]}
    selected=[]
    for name,rs in groups.items():
        selected.extend((name,r) for r in sorted(rs,key=lambda r:hashlib.sha256(('repeat-review-v1:'+r['problem_sha256']).encode()).hexdigest())[:4])
    save(out/'protocol.json',dict(input_sha256=sha(src),selection='Four per capped/not-capped stratum, SHA256(repeat-review-v1:problem_sha256) ascending; earliest repeat >=16 tokens per question; fixed before content reading.',
        populations={k:len(v) for k,v in groups.items()},selected_train_indices=[r['train_index'] for _,r in selected],
        rubric=['mechanical_restatement: repeats established material with no visible new inference in local context',
        'verification_or_reuse: repeated statement supports a new check, application, or inference',
        'uncertain: context insufficient or mixed'],
        limits='Single assistant qualitative review of eight stratified cases; not independent human labels, prevalence estimate, proof of correctness, or GPU efficacy evidence.'))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tf:raw=tf.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()==data['source_sha256'];gens=[json.loads(s) for s in raw.splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');cases=[]
    for stratum,r in selected:
        rep=r['repeats'][0];g=gens[r['question']];assert g['train_index']==r['train_index'];ids=g['token_ids'];contexts=[]
        for kind,index in [('first',rep['first_step']),('repeat',rep['step'])]:
            s=steps[index];a=max(0,s['start']-220);b=min(r['thinking_tokens'],s['stop']+300)
            contexts.append(dict(kind=kind,step=index,start=s['start'],stop=s['stop'],confidence=s['confidence'],
                before=dec.decode(ids[a:s['start']]),target=dec.decode(ids[s['start']:s['stop']]),after=dec.decode(ids[s['stop']:b])))
        cases.append(dict(stratum=stratum,train_index=r['train_index'],problem_sha256=r['problem_sha256'],problem=g['problem'],thinking_tokens=r['thinking_tokens'],
            total_repeat_steps=len(r['repeats']),contexts=contexts))
    save(out/'cases.json',dict(cases=cases));print(json.dumps(dict(cases=len(cases),output=str(out)),ensure_ascii=True))

if __name__=='__main__':main()
