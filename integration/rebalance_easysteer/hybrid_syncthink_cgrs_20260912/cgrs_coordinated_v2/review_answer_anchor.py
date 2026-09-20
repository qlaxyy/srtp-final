"""Independently recount the offline audit; no promotion from a single parent."""
import json
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    out=HERE/'answer_anchor_20260918_cpu';inp=read(out/'input.json');graded=read(out/'graded_result.json')
    assert graded['input_sha256']==sha(out/'input.json')
    assert graded['grader_sha256']==inp['grader_sha256'];assert len(graded['rows'])==len(inp['rows'])==500
    steps=read(ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/steps.json')
    selected=[];parents=[];tails=[];flips=[];candidate_conf=[]
    for r,g in zip(inp['rows'],graded['rows']):
        for k in ('question','train_index','problem_sha256','thinking_tokens'):assert r[k]==g[k]
        assert len(r['boxes'])==len(g['boxes'])
        for b,c in zip(r['boxes'],g['boxes']):
            for k,v in b.items():assert c[k]==v
        good=[b for b in g['boxes'] if b['correct']];anchor=good[0]['token_stop'] if good else None
        assert g['first_correct_box_stop']==anchor
        after=[s for s in r['repeats'] if anchor is not None and s['start']>=anchor]
        assert len(after)==g['repeat_steps_after_correct_box']
        assert sum(s['stop']-s['start'] for s in after)==g['repeat_tokens_after_correct_box']
        selected.extend(s['step'] for s in after)
        candidate_conf.extend(s['confidence'] for s in after)
        if after:parents.append(r['train_index'])
        if anchor is not None:tails.append(r['thinking_tokens']-anchor)
        if g['boxes'] and not g['boxes'][0]['correct'] and good:flips.append(r['train_index'])
    total=sum(r['thinking_tokens'] for r in inp['rows'])
    result=dict(status='not_promoted_to_GPU',reason='One parent question cannot support a general steering direction; retain frozen L27.',
        questions=500,questions_with_box=sum(bool(r['boxes']) for r in inp['rows']),questions_with_correct_box=len(tails),
        first_box_wrong_later_correct=flips,selected_repeat_steps=len(selected),selected_parent_train_indices=parents,
        selected_repeat_tokens=sum(steps[i]['stop']-steps[i]['start'] for i in selected),
        fraction_of_all_thinking_tokens=sum(steps[i]['stop']-steps[i]['start'] for i in selected)/total,
        selected_mean_confidence=float(np.mean(candidate_conf)),
        tokens_after_first_correct_box_quantiles=np.quantile(tails,[0,.5,.9,1]).tolist(),
        selected_steps=selected,inputs_sha256={n:sha(out/n) for n in ('input.json','coverage.json','graded_result.json')},
        new_generation=0,new_forward=0,grader_CPU_seconds=graded['seconds'],
        limitations=['Only explicit balanced boxed answers inside thinking were examined; unboxed answers are not covered.',
        'Training gold is used only for offline labels, unavailable to an inference-time gate.',
        'Correct intermediate answer and exact repetition are proxies, not proof of completed reasoning.',
        'Observed token share is descriptive coverage, not a predicted compression or an upper bound for other methods.'])
    save(out/'decision.json',result);print(json.dumps({k:v for k,v in result.items() if k!='selected_steps'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
