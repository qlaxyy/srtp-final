"""Freeze the native observation design; this does not launch or implement it."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from audit_sampling_order_cpu import distributions,HERE,ROOT

def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    torch.set_num_threads(2);out=HERE/'sampling_order_cpu_20260917'
    r,pre,post=distributions(.7*torch.log(torch.tensor([[.96,.04]],dtype=torch.float32)),[0])
    assert torch.isfinite(r).sum()==1 and torch.isfinite(pre).sum()==2
    assert torch.equal(torch.softmax(r,-1),torch.softmax(post,-1))
    toy={name:torch.softmax(v.float(),dim=-1).tolist()[0] for name,v in [('R',r),('RC14_order',pre),('support_preserving_order',post)]}
    source=ROOT/'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/sample/sampler.py';text=source.read_text(encoding='utf8')
    assert text.index('processed_logits = apply_top_k_top_p')<text.index('processed_logits = after_filter')<text.index('sampled = gumbel_sample')
    contract=dict(classification='New sampling-order hypothesis, not a validated optimization; no live sampler/controller changed',
        adverse_example=dict(target_column=0,probabilities=toy,meaning='Original nucleus contains only reflection. Candidate cannot choose an alternative, while originalRC14 admits one.'),
        change='Keep nucleus computed from ReBalance logits before lexical penalty. For surviving14 entries, use exactly original BF16 raw subtraction then temperature rounding. Other entries and excluded -inf unchanged.',
        preserved=['14 IDs','ln2 penalty','negative-R gate','R pre-lexical raw maximum-probability confidence','actual accepted-token boundary updates','temperature.7 and top_p.95','no added sampler draw or model forward'],
        interface=dict(placement='B-line Adapter opt-in mode using existing original_sampler(...,after_filter=callback); no S64 activation or shared-vLLM edits.',
            call_state='Immutable per-call slot-mapped mask and14 adjusted values; no global pending ticket. Replace only finite active target logits, release captures after call.',
            default_off='Do not pass callback or allocate buffers. Preserve original subtraction branch.',
            reset='No new request history. Existing opening/thinking/accepted clock; reject competing callbacks, greedy, other top_k/min_p/penalties, speculation and unsupported preemption.',
            precision='Do not replace rounded subtract-then-temperature with a simple post-temperature ln2/T subtraction.',
            backend='Seed42 already selects non-FlashInfer Gumbel path. Native invariants still required.'),
        limitations=['Scores/probabilities do not establish correctness or necessary correction.','Post-filter variant may lose compression on concentrated reflection nuclei.','Old7B captures contain zero actual eligible openings, so empirical frequency is unknown.'],
        cpu_repairs=['Primary-source annotations postponed for localPython3.9; filter body unchanged.','Adverse one-token toy compares normalized probabilities, not absolute logits (which differ by an additive constant).'],
        gpu_calls=0)
    save(out/'mechanism_contract.json',contract)
    rawroot=ROOT/'.codex_work/wsc_native_long_complete_20260917';run=next((rawroot/'results').rglob('engineering_gate.json')).parent
    oldplan=json.loads((run/'plan.json').read_text());reference=run/'RC14_wsc_shadow/result.json'
    targets=[]
    for r in json.loads(reference.read_text())['records']:
        c=np.load(run/'RC14_wsc_shadow'/f"{r['train_index']}_control.npy")
        positions=np.where((c[:,0]<0)&np.isfinite(c[:,1])&(c[:,2]==1)&(c[:,3]==1))[0][:2].tolist()
        assert len(positions)==2
        row=next(x for x in oldplan['rows'] if x['train_index']==r['train_index'])
        targets.append(dict(row,positions=positions))
    cap=1+max(p for r in targets for p in r['positions']);assert cap==201
    save(out/'native_observation_spec.json',dict(phase='Specification only; online observer/runner not yet implemented or launched',
        model='DeepSeek-R1-Distill-Qwen-1.5B',dataset='MATH Train8 already exposed engineering IDs',rows=targets,
        arms=['unchanged RC14 with read-only scratch distribution comparisons'],accuracy_answers=0,observations=16,
        max_tokens_per_request=cap,maximum_generated_prefix_tokens=cap*8,seed=42,temperature=.7,top_p=.95,
        runtime=dict(BF16=True,eager=True,async_scheduling=False,prefix_cache=False,chunked_prefill=False,max_num_seqs=32,max_num_batched_tokens=32768),
        assets=oldplan['assets'],reference_result_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        selection='First2 actual RC14 eligible positions by saved control order, not selected by logits, labels or candidate outcomes.',
        budget='Proposed generation5-20s plus startup20-40s; hard120s. Fix executable and verify timing before launch.',
        capture='Before live Adapter mutation, copy targeted logits/masks. Compare native pre-penalty top-p, originalRC14 and support-preserving alternative on scratch tensors only. Do not call an extra sampler or mutate live state.',
        record=['support sets','removed/retained14 IDs','reflection mass','total variation','all-reflection nuclei','per-call cost'],
        engineering_gate='All8 full201 prefixes and R history must match saved native reference; all16 fixed sites eligible; native p1/noop invariants pass. Any drift stops interpretation.',
        research_gate='Retained reflection support at actual eligible sites in at least2 distinct questions before proposing efficacy screening. Report concentration traps separately; no threshold search.',
        limits='201-token engineering truncations must not be graded or presented as compression/cap results. No full benchmark or7B generation. No active candidate intervention.'))
    print(json.dumps(toy));print('Observation specification:8 requests,201 tokens each,16 predetermined sites; not run.')

if __name__=='__main__':main()
