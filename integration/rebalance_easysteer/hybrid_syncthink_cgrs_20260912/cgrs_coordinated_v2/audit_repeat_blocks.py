"""Descriptive block audit fixed after eight-case sentence review, no tuning."""
import hashlib,json,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'repeat_content_review_20260918_cpu'
    protocol=dict(min_steps=3,min_tokens=64,selection='Per question earliest completed nonoverlapping exact normalized block of three adjacent stored steps and at least 64 tokens; first matching earlier block. No removal of short steps. All 500 trajectories, no length filter.',
        timing='Criterion chosen after the eight sentence cases, before block counts; exploratory, not independent validation.',
        scope='CPU coverage and case contexts only. Completion state is available, not a causally proven pre-loop trigger.')
    save(out/'block_protocol.json',protocol)
    source=HERE/'answer_anchor_20260918_cpu/input.json';base=read(source)
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer';steps=read(cache/'steps.json')
    byq=[[] for _ in range(500)]
    for i,s in enumerate(steps):byq[s['question']].append((i,s))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tf:raw=tf.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()==base['source_sha256'];gens=[json.loads(s) for s in raw.splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json');events=[]
    for q,r in enumerate(base['rows']):
        ids=gens[q]['token_ids'];ss=[(i,s) for i,s in byq[q] if s['stop']<=r['thinking_tokens']]
        texts=[' '.join(dec.decode(ids[s['start']:s['stop']]).lower().split()) for _,s in ss];seen={}
        for j in range(len(ss)-2):
            key=tuple(texts[j:j+3]);token_count=sum(s['stop']-s['start'] for _,s in ss[j:j+3])
            if token_count<64:continue
            if key in seen and seen[key]+3<=j:
                first=seen[key];first_ss=ss[first:first+3];repeat_ss=ss[j:j+3];start=repeat_ss[0][1]['start'];end=repeat_ss[-1][1]['stop']
                events.append(dict(question=q,train_index=r['train_index'],problem_sha256=r['problem_sha256'],
                    capped=r['thinking_tokens']==16000,thinking_tokens=r['thinking_tokens'],
                    first_steps=[i for i,_ in first_ss],repeat_steps=[i for i,_ in repeat_ss],
                    first_end=first_ss[-1][1]['stop'],repeat_start=start,repeat_end=end,
                    first_confidence=sum(s['confidence'] for _,s in first_ss)/3,
                    repeat_confidence=sum(s['confidence'] for _,s in repeat_ss)/3,
                    block_text=dec.decode(ids[start:end]),before=dec.decode(ids[max(0,start-180):start]),
                    after=dec.decode(ids[end:min(end+240,r['thinking_tokens'])])))
                break
            if key not in seen:seen[key]=j
    reviewed=read(out/'cases.json')['cases'];lookup={r['train_index']:r for r in events}
    summary=dict(questions=500,questions_with_block=len(events),capped_with_block=sum(r['capped'] for r in events),
        noncapped_with_block=sum(not r['capped'] for r in events),
        reviewed_question_overlap=[r['train_index'] for r in reviewed if r['train_index'] in lookup],
        events=events,source_sha256=sha(source),steps_sha256=sha(cache/'steps.json'),script_sha256=sha(__import__('pathlib').Path(__file__)),
        limitations='Events selected by completed blocks and inspected future context; neither labels nor timings certify safe online suppression. No fit, generated answer or forward.')
    save(out/'blocks.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},indent=2))
    print(json.dumps([lookup[r['train_index']] for r in reviewed if r['train_index'] in lookup],ensure_ascii=True,indent=2))

if __name__=='__main__':main()
