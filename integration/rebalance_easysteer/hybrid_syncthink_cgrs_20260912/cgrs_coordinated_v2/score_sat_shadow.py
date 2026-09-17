"""Score captured thinking prefixes after native generation has exited.

No causal correction-value or accuracy estimates. Empty/invalid requests remain
in the coverage report. Use existing environment only, no download or install.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from engineering import ROOT, read, save, sha
from sat_process import ProcessFeatures, ProtectionState, make_gru


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();start=time.monotonic()
    gate=read(args.run/'engineering_gate.json');plan=read(args.run/'plan.json')
    assert gate['passed'] and gate['plan_sha256']==sha(args.run/'plan.json')
    for name,digest in plan['sat_asset_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name,True)==digest,name
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModel
    base=ROOT/'.codex_work/sat_compatibility_20260917'
    tok=AutoTokenizer.from_pretrained(plan['assets']['model_path'],local_files_only=True)
    etok=AutoTokenizer.from_pretrained(base/'Gte-Small',local_files_only=True)
    config=AutoConfig.from_pretrained(base/'Gte-Small',local_files_only=True)
    # Avoid arbitrary pickle loading even on an older Transformers installation.
    encoder=AutoModel.from_config(config)
    weights=torch.load(base/'Gte-Small/pytorch_model.bin',weights_only=True,map_location='cpu')
    # Older BERT saves a nonpersistent position_ids buffer; validate before drop.
    key='embeddings.position_ids'
    if key in weights and key not in encoder.state_dict():
        assert torch.equal(weights[key],torch.arange(512).reshape(1,512))
        weights.pop(key)
    encoder.load_state_dict(weights,strict=True)
    encoder=encoder.eval().float().to(args.device)
    prm=make_gru(base/'step_seq_prm_gte_small_logits_gru_best_psr2.pt',args.device)
    stats=read(ROOT/'.codex_work/literature_broadening_20260917/sat/zstats.json')
    records=read(args.run/'RC14_sat_shadow/result.json')['records'];all_steps=[];report=[]
    for r in records:
        proc=ProcessFeatures(stats);steps=[];thinking=True
        path=args.run/'RC14_sat_shadow'/f"{r['train_index']}.npz"
        with np.load(path,allow_pickle=False) as trace:
            assert trace['selected'].tolist()==r['token_ids']
            for values,ids,selected in zip(trace['values'],trace['ids'],trace['selected']):
                selected=int(selected)
                if selected==151649:thinking=False
                elif selected==151648:
                    proc.fail('unexpected_second_think_start')
                elif thinking:
                    step=proc.accept(values,ids,selected,tok.decode([selected],skip_special_tokens=True))
                    if step is not None:steps.append(step)
        report.append(dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],
            invalid_reason=proc.invalid_reason,complete_steps=len(steps),unscored_tail_tokens=len(proc.rows),
            capture_sha256=sha(path)))
        all_steps.append(steps)
    startup=time.monotonic()-start;score_start=time.monotonic()
    with torch.inference_mode():
        for request,steps in zip(report,all_steps):
            guard=ProtectionState();hidden=None;scored=[]
            for offset in range(0,len(steps),64):
                group=steps[offset:offset+64]
                batch=etok([s['text'] for s in group],padding=True,truncation=True,max_length=512,return_tensors='pt').to(args.device)
                output=encoder(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1).to(output.dtype)
                emb=torch.nn.functional.normalize((output*mask).sum(1)/mask.sum(1).clamp_min(1e-9),dim=-1)
                assert emb.shape==(len(group),384) and torch.isfinite(emb).all()
                features=torch.tensor(np.stack([s['features'] for s in group]),device=args.device)[None]
                logits,hidden=prm(features,emb[None],hidden)
                scores=torch.sigmoid(logits[0]).cpu().tolist()
                for step,score in zip(group,scores):
                    scored.append(dict(start=step['start'],end=step['end'],text=step['text'],score=score,
                                       proposed_veto_after_step=guard.update(score)))
            request['steps']=scored
            request['veto_disabled_after_invalid']=bool(request['invalid_reason'])
    if args.device=='cuda':torch.cuda.synchronize()
    save(args.run/'sat_shadow_scores.json',dict(records=report,startup_seconds=startup,
        scoring_seconds=time.monotonic()-score_start,wall_seconds=time.monotonic()-start,
        actual_interventions=0,scope='Shadow engineering only; scores do not establish correction benefit',
        encoder_contract='BERT float32, max_length512, masked mean and L2 normalization; native parity still requires audit',
        passed=any(r['complete_steps']>0 and r['invalid_reason'] is None for r in report)))


if __name__=='__main__':main()
