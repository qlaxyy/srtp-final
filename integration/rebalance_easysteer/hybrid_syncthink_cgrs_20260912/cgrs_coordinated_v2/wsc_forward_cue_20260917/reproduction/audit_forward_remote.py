import json,sys,hashlib,io,time
from pathlib import Path
import numpy as np
root=Path('/root/autodl-tmp/projects/hybrid_wsc_forward_cue_20260917')
b=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(b))
from audit_wsc_weights_cpu import TensorOnlyUnpickler
from score_wsc_shadow import score_capture
from transformers import AutoTokenizer
run=Path('/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/wsc_forward_cue_20260917/wsc_forward_cue8_cap16000_run1_20260917')
plan=json.loads((run/'plan.json').read_text());arm=run/'RC14_wsc_forward_cue'
ref=Path(plan['long_reference_root']);oldarm=ref/'RC14_wsc_shadow'
for rel,digest in plan['long_reference_sha256'].items():assert hashlib.sha256((ref/rel).read_bytes()).hexdigest()==digest
new=json.loads((arm/'result.json').read_text());old=json.loads((oldarm/'result.json').read_text())
cue=json.loads((arm/'cue_report.json').read_text());slots=json.loads((arm/'export.json').read_text())['slots']
requests=json.loads((arm/'requests.json').read_text());slot_of={v['train_index']:slots[k] for k,v in requests.items()}
weights=TensorOnlyUnpickler(io.BytesIO((root/plan['probe_path']).read_bytes())).load()['model_state_dict']
w=weights['weight'].numpy().reshape(-1);bias=float(weights['bias'][0])
tok=AutoTokenizer.from_pretrained(plan['assets']['model_path'],local_files_only=True)
rows=[];scores={};started=time.monotonic()
def first_diff(a,z):
    n=min(len(a),len(z));return next((i for i in range(n) if a[i]!=z[i]),n if len(a)!=len(z) else None)
for a,z in zip(new['records'],old['records']):
    i=a['train_index'];assert i==z['train_index']
    c=cue['slots'][str(slot_of[i])];start=c['first_forced']
    ids=a['token_ids'];refids=z['token_ids'];prefix=start if start is not None else len(ids)
    token_diff=first_diff(ids[:prefix],refids[:prefix])
    control=np.load(arm/f'{i}_control.npy');refcontrol=np.load(oldarm/f'{i}_control.npy')
    n=min(prefix,len(control),len(refcontrol))
    eq=np.all((control[:n]==refcontrol[:n])|(np.isnan(control[:n])&np.isnan(refcontrol[:n])),axis=1)
    control_diff=int(np.where(~eq)[0][0]) if not eq.all() else (n if n<prefix else None)
    h=np.load(arm/f'{i}_R_history.npy');rh=np.load(oldarm/f'{i}_R_history.npy')
    assert hashlib.sha256(h.tobytes()).hexdigest()==a['R_history_sha256']
    nh=min(prefix,len(h),len(rh));r_match=np.array_equal(h[:nh],rh[:nh],equal_nan=True) and nh==prefix
    assert all(ids[e['position']]==e['token'] for e in c['forced'])
    if start is not None:assert ids[start:start+len(plan['cue_ids'])]==plan['cue_ids']
    with np.load(arm/f'{i}.npz') as data:scored=score_capture(data,plan['boundary_ids'],w,bias)
    events=scored['events'];assert len(events)==len(c['events'])
    max_score_diff=max((abs(e['score']-f['score']) for e,f in zip(events,c['events'])),default=0)
    assert max_score_diff<=plan['score_tolerance']
    assert all(e['position']==f['position'] and e['would_trigger']==f['would_trigger'] for e,f in zip(events,c['events']))
    scores[str(i)]=scored
    text=tok.decode(ids,skip_special_tokens=False);reftext=tok.decode(refids,skip_special_tokens=False)
    row=dict(train_index=i,problem_sha256=a['problem_sha256'],first_forced=start,cue_tokens=len(c['forced']),
        token_prefix_first_diff=token_diff,control_prefix_first_diff=control_diff,R_prefix_equal=r_match,
        prefix_eligible=token_diff is None and control_diff is None and r_match,
        online_offline_max_score_diff=max_score_diff,total_tokens=len(ids),reference_total_tokens=len(refids),
        thinking_tokens=ids.index(151649) if 151649 in ids else len(ids),
        reference_thinking_tokens=refids.index(151649) if 151649 in refids else len(refids),
        finish_reason=a['finish_reason'],reference_finish_reason=z['finish_reason'],
        answer_text=text.split('</think>')[-1],reference_answer_text=reftext.split('</think>')[-1],text=text)
    rows.append(row)
report=dict(rows=rows,generation_seconds=new['generation_seconds'],reference_generation_seconds=old['generation_seconds'],
    audit_seconds=time.monotonic()-started,cue_report=cue,
    all_prefixes_eligible=all(r['prefix_eligible'] for r in rows),
    scope='Exposed engineering comparison only. Rows with pre-intervention drift cannot support a causal paired effect; no statistical confirmation or synergy claim.')
with (run/'active_audit.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
with (run/'active_scores.json').open('x') as f:json.dump(dict(results=scores),f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','cue_report')}))
for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ('text','answer_text','reference_answer_text')}))
