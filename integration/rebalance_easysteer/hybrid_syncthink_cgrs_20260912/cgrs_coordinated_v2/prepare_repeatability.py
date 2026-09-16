"""Prepare only; no SSH, GPU allocation, model import or new question selection."""
from engineering import ROOT,HERE,read,save,sha
from repeatability import ARMS,SEEDS,schedule,validate

def main():
    oldpath=HERE/'first_reflection_screen100_20260914_r2/plan.json';old=read(oldpath)
    out=HERE/'history_repeatability100x3_20260916';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text whitespace=cr-at-eol\n')
    prerequisites=[oldpath,HERE/'first_reflection_screen100_20260914_r2/results/analysis.json',HERE/'frozen_prefix_layout8_20260916/results/audit.json']
    plan=dict(candidate_kind='history_repeatability',phase='exposed_training_repeatability',arms=list(ARMS),seeds=list(SEEDS),schedule=schedule(),
        primary_candidate='RChistory',rows=old['rows'],assets=old['assets'],runtime=old['runtime'],source_sha256=dict(old['source_sha256']),
        prior_plan_sha256=sha(oldpath),prerequisite_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in prerequisites},engineering_evidence=old['engineering_evidence'],
        run_ids=dict(screen='cgrs_history_repeatability100x3_20260916'),arm_seconds=dict(screen=900),process_seconds=dict(screen=8400,grade=900),
        gpu_authorized=False,confirmation200_used=False,
        estimates=dict(generation_answers=900,max_generated_tokens=14400000,minutes=[75,90],hard_generation_minutes=140,grade_hard_minutes=15,
            basis='Same100 prior R/RC14/history generation538.497/501.096/456.816 seconds; three new seeds estimate4489.23 seconds plus one model startup and grading; long-tail variability remains.'),
        execution='One model load, continuous batching32/4096, same deterministic prompt order per seed across arms; rotated arm order across seeds. No fixed-layout or per-token identity claim. No teacher forcing, probes, thresholds or controller change.',
        decision=dict(primary='RChistory vs RC14',secondary='RChistory vs R; RC14 vs R descriptive',accuracy_margin_pp=2,
            point='Against BOTH comparators: average accuracy loss <=2pp, both mean lengths strictly lower, total caps no more; both lengths lower in at least2of3 seeds against each. Otherwise no expansion of this candidate.',
            uncertainty='10000 paired question-cluster bootstrap conditional on3seeds and crossed question+seed bootstrap; require lower accuracy CI >=-2pp and upper both token-change CIs <0 in BOTH schemes before interval-supported wording; 3seed estimate remains weak.',
            next='Point-promising only warrants proposing independent confirmation; no automatic promotion or extra run. No interventions-only factorial, so no strict synergy claim.'),
        stop=['asset/code/plan/prerequisite/receipt mismatch or busy GPU','native CPU gate fails','OOM, exception, arm900s or process8400s; keep partials and stop entire batch','no seed replacement, early efficacy stopping, runtime retuning, automatic retry or expansion'],
        limitations=['Reuse already exposed100 MATH training questions: repeatability, not independent confirmation or full-test evidence.',
            '300 question-seed observations per arm are only100 unique questions; never treat them as300 independent questions.',
            'Three seeds sparsely sample decoding variability. Batch layout sensitivity remains within deployed workflow; no exact causal isolation of each lexical intervention.',
            'Order rotation mitigates but does not eliminate timing drift; generation includes inline controller cost and recorded checkpoint I/O.',
            'No U or standalone CGRS arms: cannot claim synergy or three-method superiority. Frozen tests and calibration untouched.'])
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    code=HERE.relative_to(ROOT).as_posix();rel=out.relative_to(ROOT).as_posix();result='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    gate=(HERE/'first_reflection_20260914/results/engineering_gate.json').relative_to(ROOT).as_posix()
    plan['commands']=dict(screen=f'timeout --signal=TERM --kill-after=10s 8400s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {code}/narrow_runner.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --phase screen --engineering-gate {gate} --output-root {result} --gpu-authorized',
        grade=f'timeout --signal=TERM --kill-after=10s 900s /root/autodl-tmp/venvs/rebalance/bin/python {code}/repeatability_grade.py --output {result}/{plan["run_ids"]["screen"]}')
    plan['deployment']=dict(worktree='/root/autodl-tmp/projects/hybrid_cgrs_repeatability100x3_20260916',environment='Reuse existing vLLM and isolated ReBalance grader environments; hash-checked compiled libraries; no installs or shared checkout modifications')
    validate(plan);save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,scope='history_repeatability_100x3x3_only',authorized_phases=['screen'],reuse_exposed_rows=True,plan_sha256=sha(out/'plan.json'),execution_commit=None))
    save(out/'data_usage.json',dict(status='already_exposed_7b_training100_reuse',confirmation200_used=False,new_questions=0,rows=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],purpose='three new seed repeatability; not independent confirmation') for r in plan['rows']]))
    print(sha(out/'plan.json'))

if __name__=='__main__':main()
