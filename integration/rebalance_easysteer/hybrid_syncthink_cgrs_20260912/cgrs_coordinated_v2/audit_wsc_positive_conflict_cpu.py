"""Causal availability of positive-R cancellation from saved native observations."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def audit_events(events, coefficients):
    active=False;spent=False;eligible=[];alarm_signs=[];closed_at=None
    for e,c in zip(events,coefficients):
        # ReBalance at this boundary preceded this boundary's WSC observation.
        if active and c>0:eligible.append(e['position'])
        if e['would_trigger']:
            assert not spent
            spent=True;active=True;alarm_signs.append(dict(position=e['position'],coefficient=float(c)))
        elif active and e['score']<=.5:
            active=False;closed_at=e['position']
    return dict(alarm_sites=alarm_signs,causal_positive_sites=eligible,
                first_usable=eligible[0] if eligible else None,alarm_closed_at=closed_at)


def self_test():
    ev=[dict(position=i,score=s,would_trigger=f) for i,s,f in [(4,.9,True),(8,.4,False),(12,.9,False)]]
    assert audit_events(ev,[1,-1,1])['first_usable'] is None # no retroactive or reactivation
    assert audit_events(ev,[1,1,1])['causal_positive_sites']==[8] # reset known only after injection
    assert audit_events(ev,[1,0,1])['first_usable'] is None


def main():
    self_test()
    out=HERE/'wsc_positive_conflict_20260917';plan=out/'plan.json'
    raw=ROOT/'.codex_work/wsc_native_long_complete_20260917'
    run=next((raw/'results').rglob('engineering_gate.json')).parent
    scores=json.loads((raw/'scores.json').read_text())['results']
    review=json.loads((raw/'review.json').read_text())['rows'];rows=[];inputs={}
    for r in review:
        i=r['train_index'];folder=run/'RC14_wsc_shadow'
        with np.load(folder/f'{i}.npz') as cap:
            plen=int(cap['prompt_tokens']);selected=cap['selected'];input_ids=cap['input_ids']
        control=np.load(folder/f'{i}_control.npy');history=np.load(folder/f'{i}_R_history.npy')
        events=scores[str(i)]['events'];coefs=[]
        for e in events:
            p=e['position'];assert input_ids[p+1]==selected[p]
            c=control[p+1,0]
            assert c==history[plen+p] and control[p+1,3]==1
            coefs.append(float(c))
        details=audit_events(events,coefs)
        rows.append(dict(train_index=i,problem_sha256=r['problem_sha256'],scored_boundaries=len(events),
            positive_boundaries=sum(c>0 for c in coefs),negative_boundaries=sum(c<0 for c in coefs),
            zero_boundaries=sum(c==0 for c in coefs),**details))
        for suffix in ('_control.npy','_R_history.npy'):
            p=folder/f'{i}{suffix}';inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Correct the previous audit's absolute-history prefix slicing; old files immutable.
    active_raw=ROOT/'.codex_work/wsc_forward_cue_complete_20260917'
    prior=json.loads((active_raw/'active_audit.json').read_text());absolute_audit=[]
    with tarfile.open(active_raw/'raw.tar.gz') as t:
        names={Path(m.name).name:m for m in t.getmembers() if m.isfile()}
        for r in prior['rows']:
            i=r['train_index'];folder=run/'RC14_wsc_shadow'
            with np.load(folder/f'{i}.npz') as cap:plen=int(cap['prompt_tokens'])
            new_h=np.load(io.BytesIO(t.extractfile(names[f'{i}_R_history.npy']).read()))
            old_h=np.load(folder/f'{i}_R_history.npy')
            end=plen+(r['first_forced'] if r['first_forced'] is not None else r['total_tokens'])
            assert end<=min(len(new_h),len(old_h))
            equal=np.array_equal(new_h[:end],old_h[:end],equal_nan=True)
            absolute_audit.append(dict(train_index=i,prompt_tokens=plen,exclusive_absolute_end=end,equal=equal))
    result=dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),rows=rows,
        requests_with_causal_positive_site=sum(r['first_usable'] is not None for r in rows),
        passes_coverage_gate=sum(r['first_usable'] is not None for r in rows)>=2,
        interpretation='Engineering opportunity only; no accuracy or counterfactual token-saving estimate. If coverage fails, stop without widening the rule.',
        history_audit_correction=dict(previous_issue='Previous script sliced absolute R-history with a generated-token count, omitting a prompt-length tail. Token/control prefix checks were already correctly indexed.',rows=absolute_audit,
            unchanged_prefix_conclusion=all(r['equal'] for r in absolute_audit)),input_sha256=inputs,new_model_forwards=0,new_generations=0)
    with (out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
