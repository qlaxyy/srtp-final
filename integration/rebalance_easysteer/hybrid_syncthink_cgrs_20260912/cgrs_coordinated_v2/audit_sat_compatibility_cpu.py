"""Audit pinned SAT feature and GRU contracts, without running its runner.

Only synthetic logits/embeddings are used. Does not establish score quality.
Requires existing local CPU torch/numpy; never downloads or installs anything.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def audit(source_dir, asset_dir):
    source = source_dir / 'run_think_control_V5_ds_qwen.py'
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '1f5d345cedd67226380b3c8217d4fc9c79cf73c0a5dfa5d8fe9c74f3b281b419'
    tree = ast.parse(data.decode('utf8'))
    names = {'StepSeqPRM_GRU', 'update_canonical_feats_incremental'}
    nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
    assert len(nodes) == 2
    namespace = dict(globals())
    # Excludes every import, command-line parser, model loader, and main routine.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    weights = asset_dir / 'step_seq_prm_gte_small_logits_gru_best_psr2.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == '628c182133004afee4aa07077afc899d0abbf126b3630c99585d17863b29f629'
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model = namespace['StepSeqPRM_GRU'](33, 384).eval()
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(x).all() for x in state.values())
    checks = {'checkpoint_strict_load_33_plus_384': True}
    stats = json.loads((source_dir/'zstats.json').read_text())
    names_order = ['canonical_entropy','canonical_margin','canonical_z_logp',
                   'canonical_selected_rank','canonical_logprobs','canonical_logit_gap',
                   'canonical_topk_mass@5','canonical_topk_mass@10',
                   'canonical_d_entropy','canonical_d_margin','canonical_d_logp']
    assert set(stats) == set(names_order)
    rng = np.random.default_rng(20260917)
    observed = {k: [] for k in names_order}
    expected = {k: [] for k in names_order}
    for i in range(80):
        logits = np.sort(rng.normal(size=512).astype(np.float32))[::-1].copy()
        position = i % 17
        pack = {'topk_vals': torch.tensor(logits[None,:]),
                'topk_idx': torch.arange(512).reshape(1,-1)}
        namespace['update_canonical_feats_incremental'](observed, pack, position, 50)
        z = logits.astype(np.float64)
        logp = z - np.log(np.exp(z-z.max()).sum()) - z.max()
        prob = np.exp(logp)
        row = dict(canonical_entropy=float(-(prob*logp).sum()),
                   canonical_margin=float(prob[0]-prob[1]),
                   canonical_logprobs=float(logp[position]),
                   canonical_selected_rank=float(position+1),
                   canonical_logit_gap=float(z[0]-z[1]),
                   **{'canonical_topk_mass@5':float(prob[:5].sum()),
                      'canonical_topk_mass@10':float(prob[:10].sum())})
        for dst, src in [('canonical_d_entropy','canonical_entropy'),
                         ('canonical_d_margin','canonical_margin'),
                         ('canonical_d_logp','canonical_logprobs')]:
            row[dst] = row[src]-expected[src][-1] if i else 0.0
        window = (expected['canonical_logprobs']+[row['canonical_logprobs']])[-50:]
        row['canonical_z_logp'] = (window[-1]-np.mean(window))/(np.std(window)+1e-8)
        for k,v in row.items(): expected[k].append(v)
    errors = {k:float(np.max(np.abs(np.asarray(observed[k])-expected[k]))) for k in names_order}
    assert max(errors.values()) < 5e-5, errors
    checks['independent_11_features_80_tokens_max_abs_error'] = errors
    prior = observed['canonical_logprobs'][-1]
    namespace['update_canonical_feats_incremental'](observed,pack,10000,50)
    assert observed['canonical_logprobs'][-1] == prior
    checks['public_missing_top512_repeats_previous_value'] = True
    # Standardization is token-wise, then mean/max/last, in feature-major order.
    columns = []
    for k in names_order:
        a = np.asarray(expected[k], dtype=np.float32)
        if k == 'canonical_selected_rank': a = np.log1p(a)
        mean, std = stats[k]
        columns.append((a-mean)/std)
    mat = np.stack(columns,axis=1)
    steps=[]
    for left,right in [(0,11),(11,40),(40,80)]:
        seg=mat[left:right]
        steps.append(np.stack([seg.mean(0),seg.max(0),seg[-1]],axis=1).reshape(-1))
    x=torch.tensor(np.asarray(steps))[None,:,:]
    emb=torch.tensor(rng.normal(size=(1,3,384)),dtype=torch.float32)
    emb=F.normalize(emb,dim=-1)
    with torch.no_grad():
        full=model(x,emb,torch.tensor([3]))
        hidden=None; outputs=[]
        for i in range(3):
            projected=model.input_proj(torch.cat([x[:,i:i+1],emb[:,i:i+1]],dim=-1))
            y,hidden=model.gru(projected,hidden)
            outputs.append(model.out_head(y).squeeze(-1))
        stream=torch.cat(outputs,dim=1)
        prefix=model(x[:,:2],emb[:,:2],torch.tensor([2]))
    err=float((full-stream).abs().max())
    assert err<1e-5 and torch.allclose(full[:,:2],prefix,atol=1e-5,rtol=0)
    checks['incremental_gru_matches_full_prefix_max_abs_error']=err
    checks['future_step_does_not_change_prefix_score']=True
    return dict(scope='CPU synthetic compatibility only; no semantic score validation',
                checkpoint_sha256=digest,parameter_count=sum(x.numel() for x in state.values()),
                tensor_shapes={k:list(v.shape) for k,v in state.items()},checks=checks,
                gpu_used=False,new_answers=0,real_gte_embeddings_used=False,
                feature_tolerance=5e-5,gru_tolerance=1e-5,
                blocking_gaps=['Training question IDs and split not supplied in audited files',
                               'Real GTE tokenizer/weights equivalence not tested',
                               'No pre-CGRS top512 logits in old calibration logs',
                               'Code/GRU license not established by encoder model card',
                               'No evidence yet for incremental correction-value prediction'])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.source_dir,a.asset_dir)
    with a.output.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
