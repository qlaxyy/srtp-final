"""Bounded raw-layer BCC replay, explicitly enabled after CPU preflight.

Existing BF16 HF environment only. No generate(), logits, or online steering.
Use an external 600-second process timeout as well as this cooperative deadline.
"""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from prepare_bcc import read, save, sha, require


def compare(a,b,absolute=.125,relative=.002):
    a=a.astype(np.float64);b=b.astype(np.float64)
    norm=float(np.linalg.norm(a));require(norm>1e-12,'Near-zero capture')
    maximum=float(np.max(np.abs(a-b))); ratio=float(np.linalg.norm(a-b)/norm)
    require(maximum<=absolute and ratio<=relative,'Causal-prefix consistency failed')
    return dict(max_absolute=maximum,relative_l2=ratio)


def preflight(prepared):
    plan=read(prepared/'plan.json');records=read(prepared/'prefixes.json')
    require(plan['method']=='bcc-v1-fixed-controller','Wrong plan')
    require(plan['content_gate']=='passed_under_user_clarified_standard','Content gate not passed')
    require(plan['prefixes_sha256']==sha(prepared/'prefixes.json'),'Prefix hash mismatch')
    for name,digest in plan['runtime_source_sha256'].items():
        require(sha(Path(__file__).with_name(name))==digest,'Replay source changed: '+name)
    require(plan['decoder_output_layer']==20 and plan['hidden_size']==1536,'Wrong layer/shape')
    require(len(records)==76 and sum(len(p['input_ids']) for r in records for p in r['prefixes'])==66159,'Wrong scope')
    for r in records:
        require(len(r['prefixes'])==2,'Wrong arms')
        for p in r['prefixes']:
            require(len(p['positions'])==2 and 0<=p['positions'][0]<p['positions'][1]==len(p['input_ids'])-1,'Wrong positions')
            require(p['longer_input_ids'][:len(p['input_ids'])]==p['input_ids'],'Longer input not causal extension')
    return plan,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--expected-plan-sha256')
    args=p.parse_args();plan,records=preflight(args.prepared)
    require(not args.output.exists(),'Output exists; no automatic resume or overwrite')
    if not args.execute:
        print('CPU preflight passed; no model or CUDA imported');return
    require(args.expected_plan_sha256==sha(args.prepared/'plan.json'),'Explicit plan hash required')
    started=time.perf_counter();ledger=dict(status='preflight',plan_sha256=args.expected_plan_sha256,
        script_sha256=sha(Path(__file__)),completed_pairs=0,new_answers=0,input_tokens=0,engineering_input_tokens=0)
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU process present')
    for name,digest in plan['model_files_sha256'].items():require(sha(Path(plan['model'])/name)==digest,'Model file changed: '+name)
    args.output.mkdir(parents=True);save(args.output/'plan.json',plan)
    save(args.output/'ledger.json',ledger)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        ledger.update(torch=torch.__version__,transformers=transformers.__version__,status='loading')
        save(args.output/'ledger.json',ledger)
        model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval()
        require(len(model.model.layers)>20 and model.config.hidden_size==1536,'Unexpected model architecture')
        ledger['load_and_precheck_seconds']=time.perf_counter()-started
        capture={};positions=[]
        def hook(module,inputs,value):
            h=value[0] if isinstance(value,tuple) else value
            capture['h']=h[0,positions].detach().float().cpu().numpy()
        handle=model.model.layers[20].register_forward_hook(hook)
        def forward(ids,points):
            require(time.perf_counter()-started<600,'600-second replay deadline')
            positions[:]=points;capture.clear()
            with torch.inference_mode():
                result=model.model(torch.tensor([ids],device='cuda'),use_cache=False,
                    return_dict=True,output_hidden_states=False)
            del result
            h=capture.pop('h');require(h.shape==(2,1536) and np.isfinite(h).all(),'Invalid capture')
            return h
        states=np.lib.format.open_memmap(args.output/'states.npy',mode='w+',dtype=np.float32,shape=(76,2,2,1536))
        checks=[];forward_started=time.perf_counter()
        for i,r in enumerate(records):
            for j,prefix in enumerate(r['prefixes']):
                h=forward(prefix['input_ids'],prefix['positions']);states[i,j]=h
                ledger['input_tokens']+=len(prefix['input_ids'])
                if r['pair_id'] in plan['engineering_pair_ids']:
                    repeat=forward(prefix['input_ids'],prefix['positions'])
                    require(np.array_equal(h,repeat),'Repeated capture not bitwise identical')
                    longer=forward(prefix['longer_input_ids'],prefix['positions'])
                    check=compare(h,longer,plan['max_absolute_difference'],plan['max_relative_l2'])
                    checks.append(dict(pair_id=r['pair_id'],arm=prefix['arm'],repeat_exact=True,**check))
                    ledger['engineering_input_tokens']+=len(prefix['input_ids'])+len(prefix['longer_input_ids'])
            states.flush();ledger.update(status='replaying',completed_pairs=i+1,
                forward_seconds=time.perf_counter()-forward_started,total_seconds=time.perf_counter()-started)
            save(args.output/'ledger.json',ledger)
        require(len(checks)==4 and ledger['input_tokens']==66159,'Incomplete engineering controls')
        handle.remove();del model;torch.cuda.empty_cache()
        require(time.perf_counter()-started<=600,'Deadline exceeded; do not fit incomplete run')
        save(args.output/'engineering_checks.json',checks)
        ledger.update(status='completed',states_sha256=sha(args.output/'states.npy'),
            engineering_checks_sha256=sha(args.output/'engineering_checks.json'),total_seconds=time.perf_counter()-started)
        save(args.output/'ledger.json',ledger)
    except BaseException as exc:
        ledger.update(status='incomplete',error=repr(exc),total_seconds=time.perf_counter()-started)
        save(args.output/'ledger.json',ledger);raise


if __name__=='__main__':main()
