"""Freeze engineering geometry diagnostic; never launch remotely."""
from engineering import HERE,ROOT,read,save,sha
from batch_layout_diagnostic import PHASES
from run_batch_layout_diagnostic import validate_plan

def main():
    prior=HERE/'same_logits8_20260916';old=read(prior/'plan.json')
    assert read(prior/'results/summary.json')['all_checks_passed'] is True
    prompts=read(prior/'results/prompts.json')['token_ids']
    rows=old['rows']
    for r,p in zip(rows,prompts):r.update(prompt_token_ids=p,forced_token_prefix=r['RC14_token_prefix'])
    out=HERE/'frozen_prefix_layout8_20260916';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text whitespace=cr-at-eol\n')
    plan=dict(kind='frozen_prefix_layout_v1',run_id='cgrs_frozen_prefix_layout8_20260916',phases=list(PHASES),
        assets=old['assets'],runtime=old['runtime'],rows=rows,source_sha256=dict(old['source_sha256']),
        prerequisite_sha256={str(p.relative_to(ROOT).as_posix()):sha(p) for p in [prior/'plan.json',prior/'results/summary.json',prior/'results/prompts.json']},
        max_engine_steps=dict(reference8=160,repeat8=160,single1=1100,refill4plus4=200),
        phase_seconds=dict(reference8=180,repeat8=180,single1=300,refill4plus4=180),process_seconds=900,
        estimates=dict(minutes=[3,8],outputs=32,forced_output_tokens=4096,selected_logit_rows=128,storage_MiB=200),
        acceptance='Exact model input and pre-forward native R state; repeat8 logical geometry and selected logits/all native draws/confidence must match reference8 before cross-layout tests. Cross-layout numerical differences are diagnostic findings, not automatic failure.',
        stop=['hash/asset/scope mismatch or occupied GPU','real Torch CPU test failure before model loading','input/state/slot mismatch, repeat failure, preemption, OOM or exception','phase time/step cap or 900s process watchdog; preserve partial output, no automatic rerun'],
        interpretation='New reference R states on fixed saved RC14 prefix; historical states not recovered. No correctness grading, compression claim, throughput claim, threshold tuning or independent confirmation.',
        confirmation200_used=False,gpu_authorized=False,baseline_commit='a1c9c57')
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    code=HERE.relative_to(ROOT).as_posix();rel=out.relative_to(ROOT).as_posix()
    plan['command']=f'timeout --signal=TERM --kill-after=10s 900s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {code}/run_batch_layout_diagnostic.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --output-root /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912 --gpu-authorized'
    plan['deployment']=dict(old['deployment'],worktree='/root/autodl-tmp/projects/hybrid_cgrs_frozen_prefix_layout8_20260916')
    validate_plan(plan);save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,scope='layout8_four_paths_128_only',plan_sha256=sha(out/'plan.json'),execution_commit=None))
    save(out/'data_usage.json',dict(status='previously_exposed_training_engineering_only',confirmation200_used=False,
        rows=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],purpose='fixed prefix batch geometry diagnosis') for r in rows]))
    print(sha(out/'plan.json'))

if __name__=='__main__':main()
