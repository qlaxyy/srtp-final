import json,hashlib,sys,io,os
from pathlib import Path
root=Path('/root/autodl-tmp/projects/hybrid_wsc_forward_cue_20260917')
b=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
sys.path.insert(0,str(b))
from transformers import AutoTokenizer
from audit_wsc_weights_cpu import TensorOnlyUnpickler
from score_wsc_shadow import score_capture
from wsc_forward_cue import CueState
import numpy as np
p=b/'wsc_forward_cue_20260917';plan=json.loads((p/'plan_template.json').read_text())
tok=AutoTokenizer.from_pretrained(plan['assets']['model_path'],local_files_only=True)
ids=tok.encode(plan['cue_text'],add_special_tokens=False)
assert 1<=len(ids)<=64 and not set(ids)&set(tok.all_special_ids)
assert tok.decode(ids)==plan['cue_text'];plan['cue_ids']=ids
raw=(root/plan['probe_path']).read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a2c0892a807f7bf1c4d8d954fe2262f0c87b8a91e45da7c260711bba6e501feb'
obj=TensorOnlyUnpickler(io.BytesIO(raw)).load()['model_state_dict']
w=obj['weight'].numpy().reshape(-1);bias=float(obj['bias'][0])
ref=Path(plan['long_reference_root']);scores=json.loads((p/'reference_scores.json').read_text())
checks={};refs={}
for row in plan['rows']:
    i=row['train_index'];path=ref/'RC14_wsc_shadow'/f'{i}.npz'
    with np.load(path,allow_pickle=False) as data:r=score_capture(data,plan['boundary_ids'],w,bias)
    old=scores['results'][str(i)]['events']
    assert r['events']==old
    state=CueState(ids);triggers=[]
    for e in r['events']:
        if state.observe(e['score'],e['chunk_tokens'],e['position']+1):triggers.append(e['position'])
    assert triggers==[e['position'] for e in old if e['would_trigger']]
    checks[str(i)]=dict(boundaries=len(old),trigger_positions=triggers)
    for rel in (f'RC14_wsc_shadow/{i}_control.npy',f'RC14_wsc_shadow/{i}_R_history.npy'):
        refs[rel]=hashlib.sha256((ref/rel).read_bytes()).hexdigest()
for rel in ('RC14_wsc_shadow/result.json','plan.json'):
    refs[rel]=hashlib.sha256((ref/rel).read_bytes()).hexdigest()
plan['long_reference_sha256']=refs
plan['reference_scores_sha256']=hashlib.sha256((p/'reference_scores.json').read_bytes()).hexdigest()
with (p/'plan.json').open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
receipt=dict(cue_ids=ids,cue_tokens=len(ids),checks=checks,model_forward_calls=0,plan_sha256=hashlib.sha256((p/'plan.json').read_bytes()).hexdigest())
with (p/'cpu_replay_check.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
