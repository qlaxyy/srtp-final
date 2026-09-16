"""Freeze only stage1; batch-shape replay and effect confirmation stay deferred."""
from engineering import HERE,ROOT,read,save,sha
from sampler_diagnostic import capture_positions
from run_sampler_diagnostic import validate_plan


def main():
    out=HERE/'same_logits8_20260916';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text whitespace=cr-at-eol\n')
    old=read(HERE/'first_reflection_full7b_20260914/plan_pathfix.json')
    inputs=HERE/'local_diagnosis_20260916/engineering_inputs.json'
    rows=read(inputs)['rows']
    for r in rows:r['capture_positions']=capture_positions(r['first_difference_vs_RC14'])
    plan=dict(kind='same_logits_engineering_v1',run_id='cgrs_same_logits8_20260916',assets=old['assets'],
        runtime=dict(old['runtime'],max_tokens=128,ignore_eos=True),rows=rows,
        input_sha256=sha(inputs),source_sha256=dict(old['source_sha256']),
        max_engine_steps=512,max_capture_bytes=192*1024*1024,generation_seconds=180,process_seconds=600,
        stage='same-logits sampler only; does not reproduce historical prefixes or refill layouts',
        estimates=dict(minutes=[2,5],truncated_outputs=8,max_output_tokens=1024,selected_row_observations=32,
            capture_batches_max=32,extra_native_sampler_calls_max=192,extra_model_forwards_for_comparisons=0),
        accounting='128 fixed output positions per request. ignore_eos is engineering-only so every selected position is observed; outputs after EOS are not valid task answers. No grading. Instrumented time is not benchmark speed.',
        comparisons=['native repeat','history shadow','RC14 forced inactive','history forced unarmed','synthetic active RC14 against manually penalized logits'],
        stop=['model/source/plan mismatch or occupied GPU','real Torch CPU test fails before model loading','any logits/sample/seed/live-state/mapping invariant fails',
            'preemption,OOM,exception,180s generation,512 engine steps,192MiB capture payload,600s watchdog','preserve partial outputs and diagnostics; no automatic rerun or threshold changes'],
        acceptance='All32 selected row positions captured; all five same-input cases match their exact references; live R/sampler/adapter state unchanged by comparison calls;8x128 engineering outputs only',
        interpretation='A pass clears only sampler adapter invariants on current logits. It cannot prove full-batch numerical equivalence, effect, speedup, or independent confirmation. A failure stops expansion.',
        confirmation200_used=False,other_line_status='User discontinued A line; historical data exposure/frozen assets remain protected',
        gpu_authorized=False)
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    rel=out.relative_to(ROOT).as_posix();code=HERE.relative_to(ROOT).as_posix()
    plan['command']=f'timeout --signal=TERM --kill-after=10s 600s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {code}/run_sampler_diagnostic.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --output-root /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912 --gpu-authorized'
    plan['deployment']=dict(worktree='/root/autodl-tmp/projects/hybrid_cgrs_same_logits8_20260916',
        environment='reuse existing easysteer-vllm026; no install/upgrade',
        env=dict(PYTHONNOUSERSITE='1',PATH_prefix='/root/autodl-tmp/venvs/easysteer-vllm026/bin',
                 PYTHONPATH_relative=['sources/EasySteer/vllm-steer','sources/EasySteer']),
        compiled_libraries='Verify existing deployment_identity hashes then symlink read-only compiled libraries and _version.py into independent worktree; no shared checkout change')
    validate_plan(plan);save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,scope='same_logits8_128_only',
        plan_sha256=sha(out/'plan.json'),execution_commit=None))
    save(out/'data_usage.json',dict(status='already_exposed_training_engineering_reuse',confirmation200_used=False,
        rows=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],purpose='sampler engineering only') for r in rows]))
    print(sha(out/'plan.json'))


if __name__=='__main__':main()
