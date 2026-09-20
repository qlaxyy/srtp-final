"""Probe exact step repetition without asserting semantic overthinking labels."""
import json,tarfile
from pathlib import Path
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha
from review_wsc_native_cpu import Decoder

def main():
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive) as t:rows=[json.loads(s) for s in t.extractfile('generations.jsonl').read().splitlines()]
    dec=Decoder(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json')
    seen=[set() for _ in rows];repeat=[];q=[];eligible=[];position=[];confidence=[]
    for s in steps:
        i=s['question'];ids=rows[i]['token_ids'][s['start']:s['stop']]
        text=' '.join(dec.decode(ids).lower().split());long_enough=len(ids)>=16
        repeat.append(long_enough and text in seen[i]);eligible.append(long_enough)
        seen[i].add(text);q.append(i);position.append(np.log1p(s['start']));confidence.append(s['confidence'])
    repeat=np.array(repeat);eligible=np.array(eligible);q=np.array(q)
    pos=np.array(position);conf=np.array(confidence)
    vectors=dict(original=ROOT/'.codex_work/auto_code_v2_500_20260908/auto_vector.pt',
                 length=HERE/'trajectory_length_vector_20260918_run2/auto_vector.pt')
    stats={}
    # Within-question regression adjusts linear confidence and log position.
    ix=eligible;target=repeat[ix].astype(float)
    controls=np.column_stack([conf[ix],pos[ix]])
    groups=q[ix]
    def demean(a):
        a=a.copy()
        for i in np.unique(groups):a[groups==i]-=a[groups==i].mean(axis=0)
        return a
    controls=demean(controls);rt=demean(target)
    rt-=controls@np.linalg.lstsq(controls,rt,rcond=None)[0]
    for name,path in vectors.items():
        v=torch.load(path,map_location='cpu',weights_only=True).numpy().astype(np.float64)
        z=(x@v)/np.linalg.norm(v)
        rz=demean(z[ix]);rz-=controls@np.linalg.lstsq(controls,rz,rcond=None)[0]
        stats[name]=dict(correlation_exact_repeat=float(np.corrcoef(z[ix],target)[0,1]),
            correlation_after_question_confidence_log_position_adjustment=float(np.corrcoef(rz,rt)[0,1]))
    save(HERE/'trajectory_length_vector_20260918_run2/repetition_audit.json',
        dict(eligible_steps=int(eligible.sum()),exact_repeated_steps=int(repeat.sum()),
            questions_with_exact_repeat=int(len(set(q[repeat]))),stats=stats,
            definition='Repeated >=16-token step, whitespace-normalized and lowercased, within same trajectory. Descriptive proxy only; no semantic correctness or necessity labels.',
            limitations='In-sample correlations; exact repetition can be useful restatement and misses paraphrased loops. Linear adjustments are not causal identification.',
            script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(eligible=int(eligible.sum()),repeats=int(repeat.sum()),stats=stats),indent=2))

if __name__=='__main__':main()
