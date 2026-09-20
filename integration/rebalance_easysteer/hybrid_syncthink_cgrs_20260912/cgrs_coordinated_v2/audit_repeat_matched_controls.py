"""One-pass local nuisance matching on frozen training states, not efficacy."""
import json
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    out=HERE/'repeat_matched_controls_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    source=HERE/'repeat_content_review_20260918_cpu/blocks.json'
    events=read(source)['events'];base=read(HERE/'answer_anchor_20260918_cpu/input.json')
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer';steps=read(cache/'steps.json')
    protocol=dict(blocks_sha256=sha(source),max_absolute_confidence_gap=.01,max_absolute_end_token_gap=128,
        match='For each first/repeated block endpoint, select same-question stored step not marked exact repeat >=16 tokens, excluding all six target block steps. Require >=16 tokens, within thinking, confidence gap <=0.01 and position gap <=128. Minimize normalized squared gaps; tie by step index. Two distinct controls in temporal order. No replacement of failed pairs or tolerance search.',
        interpretation='Nonrepeat is a lexical negative control, not proven valid reasoning. Position/confidence balance does not remove all confounding.',
        continuation='Fewer than 30 matched parent questions: do not fit or export a direction. At least 30 allows diagnostic geometry only, not automatic GPU promotion.',
        scope='All 145 existing exposed calibration events; no new split, benchmark selection, forward or generation.')
    save(out/'protocol.json',protocol)
    byq=[[] for _ in range(500)]
    for i,s in enumerate(steps):byq[s['question']].append(i)
    matches=[];failures=[]
    for e in events:
        q=e['question'];r=base['rows'][q];repeated={s['step'] for s in r['repeats']}
        exclude=set(e['first_steps']+e['repeat_steps']);targets=[e['first_steps'][-1],e['repeat_steps'][-1]]
        lists=[]
        for target in targets:
            t=steps[target];cand=[]
            for i in byq[q]:
                s=steps[i]
                if i in repeated or i in exclude or s['stop']>r['thinking_tokens'] or s['stop']-s['start']<16:continue
                dc=abs(s['confidence']-t['confidence']);dp=abs(s['stop']-t['stop'])
                if dc<=.01 and dp<=128:cand.append(((dc/.01)**2+(dp/128)**2,i,dc,dp))
            lists.append(sorted(cand))
        combos=[(a[0]+b[0],a,b) for a in lists[0] for b in lists[1] if a[1]<b[1]]
        if not combos:
            failures.append(dict(train_index=e['train_index'],problem_sha256=e['problem_sha256'],candidates=[len(c) for c in lists]));continue
        _,a,b=min(combos)
        matches.append(dict(train_index=e['train_index'],problem_sha256=e['problem_sha256'],targets=targets,controls=[a[1],b[1]],
            absolute_confidence_gaps=[a[2],b[2]],absolute_position_gaps=[a[3],b[3]],capped=e['capped']))
    result=dict(events=len(events),matched_questions=len(matches),unmatched_questions=len(failures),matches=matches,failures=failures,
        protocol_sha256=sha(out/'protocol.json'),steps_sha256=sha(cache/'steps.json'),status='insufficient_matched_questions' if len(matches)<30 else 'diagnostic_only',
        limitations='One fixed tolerance, exploratory minimum 30 is not a power calculation. Sparse matches cannot establish absence of a repetition-specific representation. Exact nonrepeat controls may still be semantic repetitions.')
    if len(matches)>=30:
        x=np.load(cache/'layer_21.npy',mmap_mode='r')
        raw=np.array([x[m['targets'][1]].astype(float)-x[m['targets'][0]] for m in matches])
        control=np.array([x[m['controls'][1]].astype(float)-x[m['controls'][0]] for m in matches]);residual=raw-control
        d=raw.mean(0);v=residual.mean(0)
        result['geometry']=dict(raw_norm=float(np.linalg.norm(d)),residual_norm=float(np.linalg.norm(v)),
            cosine_raw_residual=float(d@v/(np.linalg.norm(d)*np.linalg.norm(v))))
    save(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('matches','failures')},indent=2))

if __name__=='__main__':main()
