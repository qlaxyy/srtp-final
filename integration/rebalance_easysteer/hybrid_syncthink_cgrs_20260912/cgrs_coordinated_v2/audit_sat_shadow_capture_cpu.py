"""Compare captured real prefix features with the pinned official SAT function.

No model forward, labels, threshold search, or new answers. Step boundaries are
the recorded scorer boundaries; this validates numerical features, not decoder
segmentation or GTE embedding equivalence.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sat_process import FEATURES, ProcessFeatures, ProtectionState


def audit(run, source):
    raw=(source/'run_think_control_V5_ds_qwen.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='1f5d345cedd67226380b3c8217d4fc9c79cf73c0a5dfa5d8fe9c74f3b281b419'
    nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,ast.FunctionDef) and n.name=='update_canonical_feats_incremental']
    assert len(nodes)==1
    scope=dict(globals());exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_sat','exec'),scope)
    stats=json.loads((source/'zstats.json').read_text())
    scores=json.loads((run/'sat_shadow_scores.json').read_text())
    maximum=0.;steps_count=0;tokens_count=0;longest_low_run=0
    for request in scores['records']:
        ours=ProcessFeatures(stats);official={k:[] for k in FEATURES};guard=ProtectionState()
        recorded={s['end']:s for s in request['steps']}
        assert recorded and request['invalid_reason'] is None
        end=max(recorded)
        with np.load(run/'RC14_sat_shadow'/f"{request['train_index']}.npz",allow_pickle=False) as trace:
            for i in range(end):
                vals,ids,selected=trace['values'][i],trace['ids'][i],int(trace['selected'][i])
                scope['update_canonical_feats_incremental'](official,dict(topk_vals=torch.tensor(vals[None]),topk_idx=torch.tensor(ids[None])),selected,50)
                step=ours.accept(vals,ids,selected,'content\n' if i+1 in recorded else 'content')
                tokens_count+=1
                if step is None:continue
                original=recorded[i+1];assert step['start']==original['start']
                matrix=[]
                for name in FEATURES:
                    x=np.asarray(official[name][step['start']:step['end']],dtype=np.float32)
                    if name=='canonical_selected_rank':x=np.log1p(np.maximum(0,x))
                    m,s=stats[name];matrix.append((x-m)/s)
                matrix=np.stack(matrix,axis=1)
                expected=np.stack([matrix.mean(0),matrix.max(0),matrix[-1]],axis=1).reshape(33)
                error=float(np.abs(expected-step['features']).max());maximum=max(maximum,error)
                assert error<5e-5,(request['train_index'],step['end'],error)
                assert guard.update(original['score'])==original['proposed_veto_after_step']
                longest_low_run=max(longest_low_run,guard.low_run);steps_count+=1
    return dict(passed=True,real_prefix_tokens_compared=tokens_count,real_steps_compared=steps_count,
        feature_max_abs_error=maximum,tolerance=5e-5,longest_low_score_run=longest_low_run,
        requires_consecutive_low_scores=5,new_answers=0,gpu_used=False,
        limitation='Recorded boundaries supplied, no independent GTE/segmentation parity or correction-value validation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.run,args.source)
    with args.output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,indent=2))
