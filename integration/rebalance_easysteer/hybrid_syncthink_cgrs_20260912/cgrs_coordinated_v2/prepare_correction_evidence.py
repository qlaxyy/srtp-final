"""Fixed repair-cue retrieval on frozen training trajectories, not semantic labels."""
import hashlib,json,re,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha
from prepare_process_review import norm
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'correction_evidence_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    pattern=r"\b(?:I (?:made a mistake|forgot|overlooked|misread)|that(?:'s| is) (?:wrong|incorrect)|this (?:is|was) (?:wrong|incorrect))\b"
    save(out/'protocol.json',dict(pattern=pattern,selection='First matching step per parent; six parents ordered by SHA256(correction-evidence-v1:normalized_problem_hash). No confidence, correctness, cap or test selection.',
        reading='Up to900 preceding tokens, selected step and next3 steps; later context cannot justify classification available at target start.',
        matching='Review next3 steps for repetition; require explicit old wrong proposition, valid replacement and rationale for correction. Compare only within-parent observed window, not global prevalence.',
        stop='Feasibility diagnosis only: no direction fitting or GPU from six parents. No adaptive resampling of this batch.',gpu_calls=0))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gs=[json.loads(l) for l in raw.splitlines()];sp=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    assert sha(sp)=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    steps=read(sp);dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json')
    by={q:[] for q in range(500)};hits={};texts={};rx=re.compile(pattern,re.I)
    for i,s in enumerate(steps):
        q=s['question'];by[q].append(i)
        text=dec.decode(gs[q]['token_ids'][s['start']:s['stop']]);texts[i]=text
        if q not in hits and rx.search(text):hits[q]=i
    hashes={q:hashlib.sha256(norm(g['problem']).encode()).hexdigest() for q,g in enumerate(gs)}
    selected=sorted(hits,key=lambda q:hashlib.sha256(('correction-evidence-v1:'+hashes[q]).encode()).hexdigest())[:6]
    views=[];identities=[]
    for q in selected:
        g=gs[q];i=hits[q];s=steps[i];following=by[q][by[q].index(i)+1:by[q].index(i)+4]
        views.append(dict(train_index=g['train_index'],problem=g['problem'],before=dec.decode(g['token_ids'][max(0,s['start']-900):s['start']]),
            target=dict(step_index=i,text=texts[i]),following=[dict(step_index=j,text=texts[j]) for j in following]))
        identities.append(dict(train_index=g['train_index'],question=q,problem_sha256=hashes[q],generation_token_sha256=hashlib.sha256(json.dumps(g['token_ids'],separators=(',',':')).encode()).hexdigest(),
            steps=[dict(step_index=j,**steps[j]) for j in [i]+following],hidden_output_layer=20,feature_position='first content token'))
    save(out/'reading_view.json',views);save(out/'identities.json',identities)
    save(out/'retrieval.json',dict(eligible_parent_count=len(hits),selected=[gs[q]['train_index'] for q in selected],
        all_candidate_parents=[dict(train_index=gs[q]['train_index'],problem_sha256=hashes[q],first_step=hits[q]) for q in sorted(hits)],protocol_sha256=sha(out/'protocol.json')))
    print(json.dumps(dict(eligible=len(hits),selected=[gs[q]['train_index'] for q in selected])))

if __name__=='__main__':main()
