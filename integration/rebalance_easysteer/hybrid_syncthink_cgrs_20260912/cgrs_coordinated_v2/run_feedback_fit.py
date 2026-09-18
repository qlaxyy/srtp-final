"""Full saved-response fixed-history gradient engineering, never optimizer steps."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    plan=json.loads((a.input/'plan.json').read_text());a.output.mkdir(exist_ok=False,parents=True)
    start=time.monotonic()
    def timeout(*_): raise TimeoutError('900s fixed gradient engineering ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan.get('process_ceiling_seconds',900))
    try:
        for name,h in plan['input_sha256'].items(): assert sha(a.input/name)==h,name
        replay=Path(plan['replay_directory']); complete=json.loads((replay/'complete.json').read_text())
        assert (complete.get('history_exact') or complete.get('native_history_recorded')) and complete['raw_confidence_exact']
        assert sha(replay/'complete.json')==plan['replay_complete_sha256']
        manifest=json.loads((replay/'scoring_capture_manifest.json').read_text())
        for r in manifest: assert sha(replay/r['file'])==r['sha256']
        assets=plan['assets']
        for n,item in assets['model_files'].items(): assert sha(Path(assets['model_path'])/n)==item['sha256'],n
        assert sha(assets['vector']['path'])==assets['vector']['sha256']
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from feedback_fixed_history import FixedHistoryScorer
        if plan.get('query_chunk'):
            from feedback_chunked_attention import install
            install(plan['query_chunk'])
        torch.manual_seed(42)
        model=AutoModelForCausalLM.from_pretrained(assets['model_path'],torch_dtype=getattr(torch,plan.get('hf_dtype','bfloat16')),
            attn_implementation=plan.get('hf_attention','sdpa'),local_files_only=True).to('cuda').eval()
        for p in model.parameters(): p.requires_grad_(False)
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        boundaries=torch.tensor(sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s),device='cuda')
        vec=torch.load(assets['vector']['path'],map_location='cuda',weights_only=True).float()
        tables={k:torch.as_tensor(v,device='cuda') for k,v in dict(np.load(a.input/'opening.npz')).items()}
        tables['candidate_ids']=tables['candidate_ids'].long()
        scorer=FixedHistoryScorer(model,vec,boundaries,tables)
        rows=json.loads((a.input/'rows.json').read_text()); reports=[]
        from analyze_feedback_geometry import min_norm_hull
        directory=Path(plan['gradient_directory'])
        receipt=json.loads((directory/'complete.json').read_text())
        assert sha(directory/'complete.json')==plan['gradient_complete_sha256']
        for r in receipt['rows']: assert sha(directory/r['file'])==r['sha256']
        pairs=plan['pairs']; grads=[]
        for q,c,r in pairs:
            grads.append(np.load(directory/(str(q)+'_'+c+'.npz'))['gradient'].astype(float)-np.load(directory/(str(q)+'_'+r+'.npz'))['gradient'].astype(float))
        g=np.stack(grads)
        parents=json.loads((a.input/'split.json').read_text())['parents']
        fit=np.array([x['split']=='fit' for x in parents]);src=np.array([x['chosen_source']=='U' for x in parents])
        guardfit=np.array([x['question']==486 for x in parents])
        objectives=np.stack([g[fit&src].mean(0),g[fit&~src].mean(0),g[guardfit][0]])
        direction,weights=min_norm_hull(objectives)
        assert np.linalg.norm(direction)>1e-9
        step=plan['step_norm']*direction/np.linalg.norm(direction)
        delta=torch.tensor(step,device='cuda',dtype=torch.float32)
        changes={}
        with torch.no_grad():
            for row in rows:
                key=row['key'];trace=dict(np.load(replay/('scoring_capture_'+key+'.npz')))
                values=scorer.score(row['prompt_token_ids'],row['token_ids'],trace,delta)
                before=np.load(directory/(key+'.npz'))['logp']
                changes[key]=float((values.double()-torch.tensor(before,device='cuda').double()).sum())
        for i,(q,c,r) in enumerate(pairs):
            actual=changes[str(q)+'_'+c]-changes[str(q)+'_'+r]
            pred=float(g[i]@step)
            reports.append(dict(question=q,chosen=c,rejected=r,actual_margin_change=actual,predicted_margin_change=pred,relative_error=abs(actual-pred)/abs(pred)))
        margins=np.array([r['actual_margin_change'] for r in reports])
        groups={}
        for role in ['fit','development_check','correctness_guard_only']:
            for source in ['U','L27']:
                mask=np.array([x['split']==role and x['chosen_source']==source and x['question']!=486 for x in parents])
                if mask.any():groups[role+':'+source]=dict(n=int(mask.sum()),mean=float(margins[mask].mean()),median=float(np.median(margins[mask])),positive=int((margins[mask]>0).sum()),minimum=float(margins[mask].min()))
        devgroups=[v for k,v in groups.items() if k.startswith('development_check:')]
        guards=[v for k,v in groups.items() if k.startswith('correctness_guard_only:')]
        accepted=all(v['mean']>0 and v['median']>0 for v in devgroups) and all(v['minimum']>0 for v in guards) and all(margins[fit&mask].mean()>0 for mask in [src,~src]) and bool(margins[guardfit][0]>0)
        save(a.output/'groups.json',groups)
        torch.save(dict(delta=delta.cpu(),engineering_only=True,accepted=accepted),a.output/'engineering_step.pt')
        scorer.close()
        save(a.output/'complete.json',dict(status='One constrained self-feedback fit with exposed development gate; not deployed or efficacy-validated',accepted=accepted,groups=groups,rows=reports,weights=weights.tolist(),step_norm=plan['step_norm'],wall_seconds=time.monotonic()-start,model_forward_count=len(rows),optimizer_steps=1,new_answer_count=0,plan_sha256=sha(a.input/'plan.json'),artifact_sha256=sha(a.output/'engineering_step.pt')))
    except BaseException:
        save(a.output/'failure.json',dict(error=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
