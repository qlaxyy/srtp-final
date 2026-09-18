"""Separate follow-up to short-window review: preserve all six full trajectories."""
import hashlib,json,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'repair_episodes_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    prior=read(HERE/'correction_evidence_20260918_cpu/identities.json')
    save(out/'protocol.json',dict(parents=[r['train_index'] for r in prior],
        provenance='Adaptive full-context follow-up of same six previously read parents, not new independent evidence; earlier short-window records untouched.',
        task='Locate explicit false assertion, first justified repair, and subsequent recheck. Keep regression to error visible. Human/assistant evidence must cite step indices; no phrase-only labels.',
        admission='No vector fit from six parents. Output readiness decision and exact feature-position interface. No confidence matching sweeps or GPU.',
        full_context='All generated text including final region preserved, but final answer correctness not used as process labels. Per-step view plus raw decoded text avoids missing omitted trailing steps.',gpu_calls=0))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gs=[json.loads(l) for l in raw.splitlines()];sp=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    assert sha(sp)=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    steps=read(sp);dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json')
    for r in prior:
        q=r['question'];g=gs[q];assert g['train_index']==r['train_index']
        rows=[dict(step_index=i,start=s['start'],stop=s['stop'],text=dec.decode(g['token_ids'][s['start']:s['stop']])) for i,s in enumerate(steps) if s['question']==q]
        save(out/(str(r['train_index'])+'.json'),dict(identity=r,problem=g['problem'],steps=rows,full_text=dec.decode(g['token_ids']),token_count=len(g['token_ids'])))
        print(json.dumps(dict(train_index=g['train_index'],steps=len(rows),tokens=len(g['token_ids']))))

if __name__=='__main__':main()
