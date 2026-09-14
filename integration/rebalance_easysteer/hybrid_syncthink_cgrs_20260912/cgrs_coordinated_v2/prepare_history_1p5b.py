"""Freeze the unchanged history candidate for fast 1.5B exposed full tests."""
from engineering import ROOT,HERE,read,save,sha


def main():
    out=HERE/'first_reflection_full1p5b_20260915'
    out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text whitespace=cr-at-eol\n')
    plan=read(HERE/'first_reflection_full7b_20260914/plan_pathfix.json')
    old=read(HERE/'full_tests_run1/plan.json')
    plan.update(model_family='1p5b',assets=old['assets'],datasets=old['datasets'],frozen_benchmarks=old['frozen_benchmarks'])
    plan['runtime']=dict(old['runtime'],gpu_memory_utilization=.9,chunked_prefill=False)
    plan['async_engineering_rows']=read(HERE/'engineering_plan.json')['rows']
    plan['async_engineering_cap']=512
    plan['run_ids']={'full':'cgrs_first_reflection_full1p5b_math500_gsm1319_20260915'}
    plan['arm_seconds']={'full':1200}
    plan['process_seconds']={'full':3300,'grade':300}
    plan['estimates']=dict(generation_minutes=[10,15],total_minutes=[12,20],generation_answers=1819,
        engineering_truncated_outputs=32,engineering_maximum_tokens=16384,maximum_full_generated_tokens=29104000)
    plan['speed_rationale']='Reuse validated1.5B RC14 runtime: async256,32768 batch tokens,.90 memory,no chunk/prefix cache; one model load including8x4 512-token engineering gate. Abort preemption, never use sync-only replay in async.'
    plan['data_scope']='Previously exposed full tests, fixed7B candidate transferred unchanged to1.5B. Only registered engineering8 reused. No training/confirmation200 consumption.'
    plan['stop']=['Source/model/vector/fit/reference mismatch or nonidle GPU before loading',
        '8x4 async gate off/shadow token and R-history mismatch,missing intervention or invalid history order',
        'Any preemption,accepted-token clock mismatch,OOM,exception,180s engineering arm,1200s dataset,3300s process',
        'Preserve partial outputs; no silent rerun or parameter tuning']
    plan['limitations']=[x for x in plan['limitations'] if 'interrupted' not in x]
    receipt=read(HERE/'full_tests_run1/collection_receipt.json')
    prefix='results/easysteer/hybrid_syncthink_cgrs_20260912/'+old['run_id']
    remote='/root/autodl-tmp/'+prefix
    plan['rc14_reference']=dict(analysis_path=remote+'/analysis.json',analysis_sha256=receipt['files'][prefix+'/analysis.json'],results={})
    for role in old['datasets']:
        suffix='/'+role+'/RCnegative/result.json'
        plan['rc14_reference']['results'][role]=dict(path=remote+suffix,sha256=receipt['files'][prefix+suffix])
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    rel=out.relative_to(ROOT).as_posix();code=HERE.relative_to(ROOT).as_posix()
    output='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    plan['commands']=dict(full=f'timeout 3300s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {code}/narrow_runner.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --phase full --engineering-gate {code}/first_reflection_20260914/results/engineering_gate.json --output-root {output} --gpu-authorized',
        grade=f'timeout 300s /root/autodl-tmp/venvs/rebalance/bin/python {code}/full_grade.py --output {output}/{plan["run_ids"]["full"]}')
    save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,authorized_phases=['full'],exposed_test_reuse_acknowledged=False,plan_sha256=sha(out/'plan.json')))
    save(out/'data_usage.json',dict(status='fixed_exposed_test_transfer',confirmation200_used=False,
        rows=[dict(dataset=role,dataset_index=r['dataset_index'],problem_sha256=r['problem_sha256'],purpose='fixed_full_test') for role,sub in plan['datasets'].items() for r in sub['rows']],
        engineering=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],purpose='async_engineering_only') for r in plan['async_engineering_rows']]))
    print(sha(out/'plan.json'))


if __name__=='__main__':main()
