"""Frozen-calibration step labels versus native control sign. No inference."""
import json,tarfile
from pathlib import Path
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha
from prepare_label_alignment import load
from review_wsc_native_cpu import Decoder

def main():
    out=HERE/'length_state_alignment_20260918_run1';out.mkdir(exist_ok=False)
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive) as t:rows=[json.loads(s) for s in t.extractfile('generations.jsonl').read().splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json')
    seen=[set() for _ in rows];first=np.full(500,np.iinfo(np.int32).max);repeats=[]
    for s in steps:
        qi=s['question'];ids=rows[qi]['token_ids'][s['start']:s['stop']]
        text=' '.join(dec.decode(ids).lower().split());repeat=len(ids)>=16 and text in seen[qi]
        repeats.append(repeat);seen[qi].add(text)
        if repeat:first[qi]=min(first[qi],s['start'])
    rep=np.array(repeats);q=np.array([s['question'] for s in steps]);pos=np.array([s['start'] for s in steps])
    c=np.array([s['confidence'] for s in steps]);v=np.array([s['variance'] for s in steps]);h=np.array([s['lexical_hit'] for s in steps])
    lens=np.array([r['thinking_tokens'] for r in audit['questions']]);params=read(ROOT/'.codex_work/auto_code_v2_500_20260908/fit.json')['parameters']
    over=(h|(c<params['q25c']))&(lens[q]>lens.mean());under=(~h&(c>params['q75c']))&(lens[q]<lens.mean())
    runtime=load('sign_audit_runtime',ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py')
    rp=runtime.ReBalanceParams(boundary_token_ids=(1,),think_start_token_id=2,think_end_token_id=3,**params)
    co=runtime.compute_rebalance_coefficient(torch.from_numpy(c).float(),torch.from_numpy(v).float(),rp).numpy()
    assert np.isfinite(co).all()
    def group(mask):
        return dict(steps=int(mask.sum()),questions=int(len(set(q[mask]))),
            repeats=int((mask&rep).sum()),before_first_repeat=int((mask&(pos<first[q])).sum()),
            positive_control=int((mask&(co>0)).sum()),negative_control=int((mask&(co<0)).sum()),
            mean_coefficient=float(co[mask].mean()),mean_confidence=float(c[mask].mean()))
    result=dict(groups={k:group(m) for k,m in dict(all=np.ones(len(q),dtype=bool),exact_repeat=rep,
        length_over=over,length_under=under,repeat_in_length_over=rep&over,
        nonrepeat_in_length_over=~rep&over).items()},
        interpretation='Native coefficient evaluated on frozen GREEDY calibration step confidence/variance, not a replay of online interventions. Each coefficient acts AFTER its completed step; association cannot prove that positive steering caused repetitions.',
        proxy='Exact repeated >=16-token step within same question, whitespace/lowercase normalized. Not semantic gold.',
        inputs=dict(steps_sha256=sha(cache/'steps.json'),script_sha256=sha(Path(__file__)),calibration_source_sha256=audit['source_sha256']))
    save(out/'result.json',result);print(json.dumps(result['groups'],indent=2))

if __name__=='__main__':main()
