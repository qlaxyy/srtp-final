"""CPU-only distribution audit; no generation, sampler draws, or controller edit."""
import ast,hashlib,json,math,time,__future__
from pathlib import Path
import numpy as np
import torch
from policy import TRIGGERS,PENALTY

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=ROOT/'sources/EasySteer/vllm-steer/vllm/v1/sample/ops/topk_topp_sampler.py'

def native_reference_filter():
    # Execute only the self-contained primary-source CPU reference function.
    tree=ast.parse(SOURCE.read_text(encoding='utf8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_top_k_top_p_pytorch')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec',flags=__future__.annotations.compiler_flag),ns)
    return ns[fn.name]

FILTER=native_reference_filter()

def tempered(x,temperature):
    return (x.float()/torch.tensor(temperature,dtype=torch.float32)).to(x.dtype)

def distributions(raw,ids,temperature=.7,top_p=.95):
    assert raw.ndim==2 and temperature>0 and 0<top_p<=1
    values=raw[:,ids];biased=raw.clone();biased[:,ids]=values-PENALTY
    before=tempered(raw,temperature);adjusted=tempered(biased,temperature)
    cutoff=torch.full((raw.shape[0],),top_p,dtype=torch.float32)
    base=FILTER(before.clone(),None,cutoff)
    pre=FILTER(adjusted.clone(),None,cutoff)
    post=base.clone()
    # Exact original BF16 subtraction-then-temperature for surviving IDs.
    # Avoid a new roundoff confound from subtracting ln2/T after rounding.
    post[:,ids]=torch.where(torch.isfinite(base[:,ids]),adjusted[:,ids],base[:,ids])
    assert torch.equal(torch.isfinite(base),torch.isfinite(post))
    return base,pre,post

def checks():
    passed=[]
    for dtype in (torch.float32,torch.bfloat16):
        raw=torch.tensor([[2.,1.,0.,-1.]],dtype=dtype)
        base,pre,post=distributions(raw,[1],top_p=1.)
        assert torch.equal(pre,post);passed.append(str(dtype)+' p1 exactly equal')
        base,pre,post=distributions(raw,[],top_p=.95)
        assert torch.equal(base,pre) and torch.equal(pre,post)
        raw=torch.tensor([[10.,9.,-20.]],dtype=dtype)
        base,pre,post=distributions(raw,[2])
        assert not torch.isfinite(base[0,2]) and torch.equal(base,post)
    raw=.7*torch.log(torch.tensor([[.94,.04,.02]],dtype=torch.float32))
    base,pre,post=distributions(raw,[1])
    assert torch.isfinite(base[0,1]) and not torch.isfinite(pre[0,1]) and torch.isfinite(post[0,1])
    assert not torch.isfinite(post[0,2])
    toy={k:torch.softmax(v.float(),dim=-1).tolist()[0] for k,v in [('R',base),('RC14_order',pre),('support_preserving_order',post)]}
    return dict(checks=passed+['no targets unchanged','excluded target not revived','toy target retained vs removed'],toy=toy,
        ideal_pairwise_odds_multiplier=math.exp(-PENALTY/.7),
        note='Exact-arithmetic odds statement before filtering; final marginal probability is not multiplied by a fixed factor.')

def main():
    torch.set_num_threads(2);start=time.monotonic();out=HERE/'sampling_order_cpu_20260917'
    plan=json.loads((out/'plan.json').read_text(encoding='utf8'));test=checks()
    src=Path(plan['capture_root']);manifest=json.loads((src/'manifest.json').read_text())
    runtime=json.loads((src/'results/runtime.json').read_text());mapping=runtime['request_mapping']
    complete=json.loads((src/'results/result.json').read_text());assert all(e['changed']==0 for e in complete['events'].values())
    requests={r['train_index']:r for r in json.loads((HERE/'same_logits8_20260916/plan.json').read_text())['rows']}
    rows=[];seen=set();inputs={};batches=0
    for path in sorted((src/'results').glob('capture_*.pt')):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==manifest['results/'+path.name]
        blob=torch.load(path,map_location='cpu',weights_only=True);inputs[path.name]=digest;batches+=1
        for j,(rid,slot,valid) in enumerate(zip(blob['requests'],blob['slots'],blob['valid'])):
            if not valid:continue
            count=int(blob['owner']['count'][slot]);key=(rid,count)
            if key in seen:continue
            seen.add(key);coef=float(blob['coefs'][slot]);mean=float(blob['prev_step_mean'][slot])
            eligible=bool(blob['owner']['opening'][slot] and blob['owner']['thinking'][slot] and math.isfinite(coef) and math.isfinite(mean) and coef<0)
            identity=dict(train_index=mapping[rid],problem_sha256=requests[mapping[rid]]['problem_sha256'],generated_position=count,eligible=eligible,coefficient=coef,raw_dtype=str(blob['logits'].dtype))
            if not eligible:rows.append(identity);continue
            raw=blob['logits'][j:j+1];base,pre,post=distributions(raw,list(TRIGGERS))
            p,q=[torch.softmax(x.float(),dim=-1)[0] for x in (pre,post)]
            diff=torch.isfinite(pre[0])!=torch.isfinite(post[0]);removed=[i for i in TRIGGERS if torch.isfinite(post[0,i]) and not torch.isfinite(pre[0,i])]
            same=not bool(diff.any());assert not same or torch.equal(pre,post)
            rows.append(dict(identity,support_difference_tokens=int(diff.sum()),retained_trigger_ids=removed,
                total_variation=float((p-q).abs().sum()/2),original_trigger_mass=float(p[list(TRIGGERS)].sum()),
                candidate_trigger_mass=float(q[list(TRIGGERS)].sum()),
                original_support=int(torch.isfinite(pre).sum()),candidate_support=int(torch.isfinite(post).sum())))
    eligible=[r for r in rows if r['eligible']];changed=[r for r in eligible if r['support_difference_tokens']]
    # Window evidence from previously frozen complete RC14 controls, distinct model/data.
    rawroot=ROOT/'.codex_work/wsc_native_long_complete_20260917'
    run=next((rawroot/'results').rglob('engineering_gate.json')).parent;window=[]
    records=json.loads((run/'RC14_wsc_shadow/result.json').read_text())['records']
    for r in records:
        c=np.load(run/'RC14_wsc_shadow'/f"{r['train_index']}_control.npy")
        flags=(c[:,2]==1)&(c[:,3]==1)&np.isfinite(c[:,1])&(c[:,0]<0)
        spans=[];length=0
        for f in [*flags,False]:
            if f:length+=1
            elif length:spans.append(length);length=0
        window.append(dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],eligible_positions=int(flags.sum()),
            windows=len(spans),multi_position_windows=sum(n>1 for n in spans),maximum_window=max(spans,default=0)))
    result=dict(plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        filter_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),checks=test,
        logits_model='DeepSeek-R1-Distill-Qwen-7B',capture_batches=batches,unique_captured_positions=len(rows),eligible_positions=len(eligible),
        changed_support_positions=len(changed),changed_support_questions=len({r['train_index'] for r in changed}),
        opportunity_gate_passed=len({r['train_index'] for r in changed})>=2,
        rows=rows,window_model='DeepSeek-R1-Distill-Qwen-1.5B',window_rows=window,input_sha256=inputs,
        cpu_seconds=time.monotonic()-start,new_model_forwards=0,new_answers=0,
        limitations=['Saved7B short engineering snapshots under R, not long RC14 or independent1.5B evidence.',
            'CPU primary-source PyTorch top-p reference; GPU uses Triton, boundary rounding/ties require native verification.',
            'Same-logits distribution differences only. No sampled outputs, correctness, token savings, or counterfactual trajectory estimate.',
            'Support preservation keeps available reflection tokens but does not prove they are needed or their probabilities sufficient.'])
    with (out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256','window_rows')},ensure_ascii=False,indent=2))
    print('changed eligible:',json.dumps(changed,ensure_ascii=False));print('windows:',json.dumps(window))

if __name__=='__main__':main()
