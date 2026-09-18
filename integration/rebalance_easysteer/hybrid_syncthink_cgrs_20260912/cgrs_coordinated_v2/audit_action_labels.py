"""Recount old apply/skip evidence without confusing action with state labels."""
import collections,json
from prepare_length_vector import HERE,ROOT,read,save,sha

def classify(a,b,ambiguous=False):
    if ambiguous:return 'grading_ambiguous'
    if any(r['finish_reason']=='length' or not r['think_closed'] for r in (a,b)):return 'censored_or_unfinished'
    if not (a['author_correct'] and b['author_correct']):return 'correctness_tradeoff_or_failure'
    dt=a['total_tokens']-b['total_tokens'];dh=a['think_tokens']-b['think_tokens']
    if dt<0 and dh<0:return 'apply_shorter_both_correct'
    if dt>0 and dh>0:return 'skip_shorter_both_correct'
    if dt==dh==0:return 'equal_lengths'
    return 'length_objectives_disagree'

def main():
    folder=ROOT/'.codex_work/boundary_ablation20_20260910';run=folder/'run2'
    summary=read(run/'summary.json');manifest=read(folder/'manifest.json');protocol=read(run/'protocol.json')
    assert protocol['manifest_sha256']==sha(folder/'manifest.json')
    assert manifest['artifacts']['model_inputs_sha256']==sha(folder/'model_inputs.json')
    for n,h in summary['hashes'].items():assert sha(run/n)==h,n
    review=read(ROOT/'integration/rebalance_easysteer/configs/boundary_failure_review6_20260910.json')
    assert review['findings']['completed_correct_choice_false_negatives']==2
    raw=[json.loads(l) for l in (run/'continuations.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(raw)==40;by={(r['case_id'],r['arm']):r for r in raw};scores={(r['case_id'],r['arm']):r for r in summary['records']}
    inputs={r['case_id']:r for r in read(folder/'model_inputs.json')}
    rows=[]
    for case in manifest['cases']:
        cid=case['case_id'];paired=[]
        for arm in ('apply','skip'):
            r=by[cid,arm];s=scores[cid,arm];ids=inputs[cid]['generated_prefix_token_ids']+r['token_ids']
            assert r['total_tokens']==len(ids)<=16000
            assert r['think_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
            for k in ('train_index','total_tokens','think_tokens','finish_reason'):assert s[k]==r[k]
            paired.append(dict(r,author_correct=s['author_correct']))
        a,b=paired;label=classify(a,b,cid=='B04')
        rows.append(dict(case_id=cid,train_index=case['train_index'],stratum=case['stratum'],
            prefix_sha256=case['prefix_sha256'],coefficient=case['projected_coefficient_float32'],
            apply_minus_skip_thinking=a['think_tokens']-b['think_tokens'],apply_minus_skip_total=a['total_tokens']-b['total_tokens'],
            apply_correct=a['author_correct'],skip_correct=b['author_correct'],label=label))
    out=HERE/'action_label_audit_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    result=dict(status='retrospective_label_feasibility_not_new_generation',counts=dict(collections.Counter(r['label'] for r in rows)),
        counts_by_sign={g:dict(collections.Counter(r['label'] for r in rows if r['stratum']==g)) for g in ('strong_negative','positive')},rows=rows,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (run/'summary.json',run/'continuations.jsonl',folder/'manifest.json',ROOT/'integration/rebalance_easysteer/configs/boundary_failure_review6_20260910.json')},
        decision='Action preference labels are conditional on the signed original displacement, one greedy continuation, and an unsteered saved prefix. They cannot be treated as semantic over/under classes or directly converted to a new steering direction.',
        limitations=['Both final answers correct does not prove full reasoning valid. B04 is quarantined for known option-mapping issue, without rewriting author scores.',
        'Positive and negative coefficients must not be pooled as one intervention. Single sampled counterfactual is not expected treatment effect.',
        'This predates L27 and has no L27 suppression; no claim of current combination benefit.',
        'A new vector would itself change the intervention used to create these labels; transport must be tested separately.'],
        new_answers=0,new_forward=0)
    save(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
