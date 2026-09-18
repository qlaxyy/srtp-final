"""Stage-stratified calibration review, without selecting on confidence or lexicon."""
import hashlib,json,tarfile,unicodedata
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def main():
    out=HERE/'process_review_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    protocol=dict(selection='Original500 parents stratified by observed thinking cap16000:6 each by SHA256(process-stage-v1:normalized_problem_hash). Require4 steps >=16 tokens. Each parent: floor(n*.25), floor(n*.5), floor(n*.75) in eligible steps.',
        labels=['valid_advancement','necessary_correction','redundant_restatement','harmful_revision','mixed_or_uncertain'],
        evidence='Use problem and previous context only. A redundant label needs explicit prior support; exact text recurrence alone is not enough. A correction needs old error and justified new proposition. Store local mathematical validity separately.',
        limits='12 parents,36 steps, stratified content diagnosis, not prevalence or training-grade gold. Labels/confidence/final grading hidden in reading view; wording and cap stratum may remain visible. Do not fit a direction from this review.',
        gpu_calls=0,new_answers=0)
    save(out/'protocol.json',protocol)
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gs=[json.loads(l) for l in raw.splitlines()];sp=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    assert sha(sp)=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    steps=read(sp);by={q:[] for q in range(500)}
    for i,s in enumerate(steps):
        if s['stop']-s['start']>=16:by[s['question']].append(i)
    lengths=[g['token_ids'].index(151649) if 151649 in g['token_ids'] else len(g['token_ids']) for g in gs]
    hashes=[hashlib.sha256(norm(g['problem']).encode()).hexdigest() for g in gs];selected=[]
    for capped in (False,True):
        pool=[q for q in range(500) if (lengths[q]==16000)==capped and len(by[q])>=4]
        selected+=sorted(pool,key=lambda q:hashlib.sha256(('process-stage-v1:'+hashes[q]).encode()).hexdigest())[:6]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');views=[];identities=[]
    for q in selected:
        g=gs[q];ids=g['token_ids'];indices=by[q];texts={i:dec.decode(ids[steps[i]['start']:steps[i]['stop']]) for i in indices}
        for stage,fraction in [('quarter',.25),('middle',.5),('late',.75)]:
            i=indices[int(len(indices)*fraction)];s=steps[i];prior=[j for j in indices if j<i and norm(texts[j])==norm(texts[i])]
            target=texts[i];assert len(target)<20000
            case_id=str(g['train_index'])+'_'+stage
            views.append(dict(case_id=case_id,problem=g['problem'],stage=stage,before=dec.decode(ids[max(0,s['start']-250):s['start']]),target=target,
                earlier_exact_match=dict(step_index=prior[0],start=steps[prior[0]]['start'],text=texts[prior[0]]) if prior else None))
            identities.append(dict(case_id=case_id,train_index=g['train_index'],question=q,problem_sha256=hashes[q],generation_token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                step_index=i,start=s['start'],stop=s['stop'],thinking_tokens=lengths[q],confidence=s['confidence'],lexical_hit=s['lexical_hit'],feature_row=i,hidden_output_layer=20))
    save(out/'reading_view.json',views);save(out/'identities.json',identities)
    print(json.dumps(dict(parents=[gs[q]['train_index'] for q in selected],steps=len(views),protocol_sha256=sha(out/'protocol.json'))))
if __name__=='__main__':main()
