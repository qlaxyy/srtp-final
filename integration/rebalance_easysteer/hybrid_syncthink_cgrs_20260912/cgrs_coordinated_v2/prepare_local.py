"""CPU-only package freeze. Does not connect, load a model, or reserve new data."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=HERE.parent/'cgrs_probe_20260913'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p,canonical=False):
    b=p.read_bytes()
    return hashlib.sha256(b.replace(b'\r\n',b'\n') if canonical else b).hexdigest()
def save(name,v):
    with (HERE/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def main():
    check=subprocess.run([sys.executable,str(HERE/'test_cpu.py')],capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    identity=read(OLD/'identity.json')
    sources={name:h for name,h in identity['source_sha256'].items()
             if name.startswith(('sources/','integration/rebalance_easysteer/eval/'))}
    for p in HERE.glob('*.py'):sources[p.relative_to(ROOT).as_posix()]=digest(p,True)
    for name,h in sources.items():assert digest(ROOT/name,True)==h,name
    rows=read(OLD/'execution_run5/plan.json')['rows']
    save('engineering_plan.json',dict(run_id='cgrs_coordinated_v2_engineering8_run1_20260913',
        phase='engineering_only',arms=['R','Roff','Rshadow','RC','Clex'],rows=rows,
        source_sha256=sources,assets=identity['assets'],
        runtime=dict(async_scheduling=True,max_num_seqs=256,max_num_batched_tokens=32768,
            dtype='bfloat16',max_model_len=32768,max_tokens=256,seed=42,temperature=.7,top_p=.95),
        controls='R original; Roff installs nothing; Rshadow computes gate but never changes logits; RC is coordinated soft lexical penalty; Clex is unconditional lexical-only ablation, not official CGRS',
        gpu_authorized=False,estimated_minutes=[2,4],process_ceiling_seconds=900,arm_ceiling_seconds=180,
        stop=['native CPU failure','source/asset mismatch','KV preemption','accepted-token clock mismatch',
              'Roff or Rshadow token/history mismatch','no gate/intervention coverage','timeout or exception'],
        command='timeout 900s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python '+
            HERE.relative_to(ROOT).as_posix()+'/engineering.py --plan '+HERE.relative_to(ROOT).as_posix()+
            '/engineering_plan.json --output-root /root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912 --gpu-authorized',
        deployment='Only after user enables/authorizes this batch: isolated copy of prior hash-checked B runtime; upload this namespace, no shared vLLM edit, no package installation. GPU gate first; not an efficacy benchmark.'))
    save('data_usage.json',dict(status='existing_engineering8_only_no_new_split_frozen',
        rows=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],
                   purpose='reused B engineering only, never independent confirmation') for r in rows],
        new_train_reserved=[],coordination='New effect-screen and confirmation IDs require fresh cross-line reconciliation before selection/freeze; prior full tests and consumed train64 excluded from new independent data.'))
    full=read(OLD/'full_tests_run1/results/summary.json')
    costs={}
    for role,r in full['datasets'].items():
        costs[role]=dict(probe_flow_seconds=r['cost']['probe_wall_seconds'],
            generation_seconds=r['generation_seconds'],historical_R_seconds=r['historical_R']['generation_seconds'],
            generation_minus_probe_seconds=r['generation_seconds']-r['cost']['probe_wall_seconds'])
    save('cpu_findings.json',dict(
        source_summary_sha256=digest(OLD/'full_tests_run1/results/summary.json'),costs=costs,
        established=['Recorded probe workflow does not account for all timing difference: subtracting it is arithmetic, not a forecast of achievable runtime.',
                     'Old should_mask does not inspect R coefficient; opposing interventions are possible by construction, not measured as a causal explanation.',
                     'Neither proposed penalty nor sign gate was selected by a test-set threshold sweep.'],
        unknown=['Whether the new R sign proxy predicts redundant reflection','Actual asynchronous speed and batch equivalence',
                 'Whether either combination beats both single components; no new generation has occurred'],
        cpu_tests=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr),
        native_torch_tests='prepared, not executed locally; torch unavailable',gpu_runs=0,new_answers=0,ssh_connections=0))
    save('implementation_spec.json',dict(
        identity='ReBalance-coordinated CGRS-inspired soft lexical adaptation; removes answer-probe certainty and is not official CGRS',
        fixed_rule='At a clean original R step opening while thinking, require finite closed-step R mean and coefficient < 0. Subtract ln(2) from the same 14 trigger logits. No R modification, probe, RNG, forced end, or answer replacement.',
        signal='Read the existing coefficient sign, not a new confidence threshold. R raw max-probability already computed before the sampler adapter; native R updates after accepted sampling. Next step reads that updated controller.',
        semantics='ln(2) halves raw trigger/non-trigger odds. Temperature 0.7 and top-p 0.95 subsequently alter probabilities and may still remove a penalized token. Finite penalty does not guarantee necessary correction survives.',
        lifecycle='Per native request slot: count, prompt length, clean opening, thinking, eligibility/change counts. Initialize on fresh admission, reset reused slots, close on think-end. Read completion metrics once; preemption and unsupported inputs fail.',
        files='Only this namespace and existing B README/handoff. Instance hooks: runner.sampler, add_requests, _remove_request; scheduler._preempt_request. Default off installs no hooks. No shared controller fork.',
        expected_benefit='Avoid sampled-probe noise/copying/pauses and conflicting positive-R plus lexical-suppression actions; permit native asynchronous scheduling and FINAL_ONLY output.',
        risks='R confidence is not truth and can be high in loops. Shared signal may be redundant. Sign gating can miss redundancy. But/Alternative still have ordinary semantic uses. Softmax distribution changes all normalized probabilities.',
        candidate_tradeoff='This candidate jointly changes signal, suppression strength and execution path; it is not an isolated timing-only ablation of v1.',
        next_effect_design=dict(status='proposal_only_no_data_frozen_or_GPU_authorized',
            scope='1.5B, new MATH training screening64, five arms U/R/Clex/RCalways/RCnegative; fixed max16000, seed42, same async settings.',
            purpose='U/R/Clex/RCalways provides factorial contrast; RCnegative versus RCalways isolates the coordination gate. Clex is only the borrowed lexical mechanism, not full CGRS.',
            estimated_minutes=[15,25],accuracy_margin_pp=2,
            report=['accuracy','thinking/total tokens','caps','wrong/right transitions','generation wall time','host/device control overhead','GPU utilization time series'],
            selection='Keep only if thought and total mean both decrease versus R, point accuracy loss <=2pp, caps do not increase. Any small reproducible saving counts. Report paired 10000-sample CI and limitations; no >5% requirement.',
            interaction='On mean token counts I=T_RCalways-T_R-T_Clex+T_U. I<0 with uncertainty and comparable accuracy supports superadditive savings on this scale. The gated candidate needs the additional direct contrast; ratios from different sets are never added.',
            confirmation='Separate untouched training data after screening; recheck both lines before freezing. Do not reuse full tests or old screen64; do not call point accuracy margin statistical noninferiority. No automatic 7B/full expansion.')))
    print('CPU PACKAGE PREPARED; NO GPU RUN')


if __name__=='__main__':main()
