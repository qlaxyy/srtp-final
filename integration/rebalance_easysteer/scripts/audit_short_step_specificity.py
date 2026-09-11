"""Test whether previous-step borrowing beats equally weighted generic shrinkage."""
import hashlib
import json
import tarfile
import numpy as np
from mechanism_candidates import ROOT, BASE, require, read, save, sha


def main():
    plan_path=ROOT/BASE/'configs/short_step_specificity_plan_20260911.json'
    plan=read(plan_path)
    out=ROOT/'.codex_work/mechanism_candidates_20260911/short_specificity.json'
    require(not out.exists(),'Do not rerun completed diagnostic')
    original=read(ROOT/BASE/'configs/mechanism_cpu_plan_20260911.json')
    folder=ROOT/original['inputs']['backup']; steps=read(folder/'steps.json')
    manifest=read(ROOT/original['inputs']['backup_manifest'])
    for name in ('steps.json','selected_layer.json'):
        require(sha(folder/name)==manifest['files'][name]['sha256'],'Changed input')
    split=read(folder/'selected_layer.json'); train=set(split['training_questions']); valid=set(split['validation_questions'])
    with tarfile.open(ROOT/original['inputs']['generations_archive']) as archive:
        with archive.extractfile('generations.jsonl') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest()==original['inputs']['generations_sha256'],'Changed answers')
        rows=[json.loads(line) for line in archive.extractfile('generations.jsonl')]
    byq={q:[] for q in range(500)}
    for s in steps:byq[s['question']].append(s)
    numerator=sum((s['stop']-s['start'])*s['confidence'] for s in steps if s['question'] in train)
    denominator=sum(s['stop']-s['start'] for s in steps if s['question'] in train)
    prior=numerator/denominator
    question_losses=[]
    for q in sorted(valid):
        probs=np.exp(rows[q]['logprobs']); previous=None; losses=[]
        for s in byq[q]:
            values=probs[s['start']:s['stop']]
            if 2<=len(values)<=8 and previous is not None:
                k=len(values)//2; n=min(len(previous),8); target=values[k:].mean()
                temporal=(values[:k].sum()+previous[-8:].sum())/(k+n)
                shrink=(values[:k].sum()+n*prior)/(k+n)
                losses.append([(temporal-target)**2,(shrink-target)**2])
            previous=values
        if losses:question_losses.append(dict(question=q,steps=len(losses),mse=np.mean(losses,axis=0).tolist()))
    temporal,shrink=np.mean([r['mse'] for r in question_losses],axis=0)
    ratio=float(temporal/shrink)
    report=dict(status='passed_specificity_requires_online_preparation' if ratio<=.95 else 'deferred_temporal_specificity_not_supported',
        plan=plan,plan_sha256=sha(plan_path,source=True),script_sha256=sha(__file__,source=True),
        train_prior_probability=prior,validation_questions=len(question_losses),
        temporal_mse=float(temporal),fixed_mean_mse=float(shrink),ratio=ratio,
        question_losses=question_losses,new_generations=0,model_forwards=0,server_connections=0)
    save(out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('plan','question_losses')},ensure_ascii=False))


if __name__=='__main__':main()
