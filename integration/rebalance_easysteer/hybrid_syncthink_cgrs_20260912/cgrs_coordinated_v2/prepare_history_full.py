"""Freeze candidate and full test inputs without executing inference."""
from pathlib import Path
from engineering import ROOT,HERE,read,save,sha


def main():
    out=HERE/'first_reflection_full7b_20260914';assert not out.exists();out.mkdir()
    (out/'.gitattributes').write_text('*.json -text whitespace=cr-at-eol\n')
    old=read(HERE/'full_7b_run1/plan.json')
    screen=read(HERE/'first_reflection_screen100_20260914_r2/plan.json')
    local=ROOT/'.codex_work/cgrs_v2_7b_run1_verified/results/easysteer/hybrid_syncthink_cgrs_20260912'/old['run_id']
    remote=Path('/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912')/old['run_id']
    ref=dict(analysis_path=str(remote/'analysis.json'),analysis_sha256=sha(local/'analysis.json'),results={})
    for role in old['datasets']:
        p=local/role/'RCnegative/result.json'
        ref['results'][role]=dict(path=str(remote/role/'RCnegative/result.json'),sha256=sha(p))
    plan=dict(schema=1,candidate_kind='first_reflection_full',phase='full_test',arms=['RChistory'],primary_candidate='RChistory',
        runtime=screen['runtime'],assets=screen['assets'],datasets=old['datasets'],frozen_benchmarks=old['frozen_benchmarks'],
        rc14_reference=ref,engineering_evidence=screen['engineering_evidence'],source_sha256=screen['source_sha256'],
        run_ids=dict(full='cgrs_first_reflection_full7b_math500_gsm1319_20260914'),arm_seconds=dict(full=4800),
        process_seconds=dict(full=9900,grade=600),estimates=dict(minutes=[35,60],generation_answers=1819,maximum_generated_tokens=29104000),
        decision=dict(fixed_before_generation=True,criterion='Per dataset versus historical R: accuracy loss<=2pp, mean thought and total tokens both lower, caps<=R. Also report same comparisons vs RC14; do not select thresholds after results.',
            uncertainty='10000 paired percentile bootstrap seed20260913. Point2pp bound distinct from nominal95% CI noninferiority. No strict causal or factorial synergy claim.',
            aggregate='Report each dataset separately; no pooled pass to hide MATH regression.'),
        attribution_limit='Prior screen59/100 diverged before own first mask including3 accuracy gains; unresolved. This evaluates complete pipeline, not direct per-question causal action.',
        data_scope='Previously exposed MATH500/GSM8K1319 tests; candidate fixed after new training100 screen. No test-set tuning, no confirmation200 usage.',
        speed_rationale='Stable32 concurrent requests,4096 batch tokens,.94memory; full continuous queue and one model load. Prior48 short-window throughput worsened8.5%; no unvalidated concurrency escalation.',
        stop=['Asset/source/protocol/prompt-capacity mismatch before model loading','Native test failure,OOM,exception,4800s dataset or9900s process ceiling',
              'Preserve finished outputs and unfinished prefixes. Do not silently retry or replace rows; no live parameter tuning'],
        limitations=['Historical U/R/RC14 controls differ in runtime and batch scheduling; no simultaneous speed benchmark',
            'Historical R MATH time includes an interrupted run and is not directly comparable',
            'CGRS-inspired history adaptation, not official implementation; no extra probe or auxiliary forward',
            'Single seed; paired question intervals omit runtime variation and pre-intervention batch divergence'])
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    rel=out.relative_to(ROOT).as_posix();prefix=HERE.relative_to(ROOT).as_posix()
    output='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    plan['commands']=dict(full=f'timeout 9900s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {prefix}/narrow_runner.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --phase full --engineering-gate {prefix}/first_reflection_20260914/results/engineering_gate.json --output-root {output} --gpu-authorized',
        grade=f'timeout 600s /root/autodl-tmp/venvs/rebalance/bin/python {prefix}/full_grade.py --output {output}/{plan["run_ids"]["full"]}')
    save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,authorized_phases=['full'],exposed_test_reuse_acknowledged=False,plan_sha256=sha(out/'plan.json')))
    save(out/'data_usage.json',dict(status='fixed_exposed_test_evaluation',confirmation200_used=False,
        rows=[dict(dataset=role,dataset_index=row['dataset_index'],problem_sha256=row['problem_sha256'],purpose='fixed_full_test') for role,sub in plan['datasets'].items() for row in sub['rows']]))
    print(sha(out/'plan.json'))


if __name__=='__main__':main()
