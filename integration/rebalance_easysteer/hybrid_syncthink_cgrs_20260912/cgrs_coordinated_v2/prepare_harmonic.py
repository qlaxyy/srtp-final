"""Freeze one controller aggregation experiment before GPU outcomes exist."""
import json,hashlib,shutil
from pathlib import Path
from prepare_label_alignment import save,sha
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    out=HERE/'harmonic_confidence_20260918';old=HERE/'type_split_20260917'
    audit=read(out/'calibration_audit.json')
    fit=read(Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908/fit.json'))
    fit.update(parameters=audit['harmonic_parameters'],version='harmonic-controller-v1',
        method='ATAR harmonic aggregation inspired adaptation of raw maximum probabilities. Frozen arithmetic-labelled raw vector/layer/amplitudes; recalibrated signal quantiles and feasible tau. Not official ATAR or CGRS.')
    save(out/'fit.json',fit)
    plan=read(old/'plan.json');plan.update(run_id='harmonic_1p5b_math500_20260918_run1',experiment_kind='harmonic_v1',
        arms=[dict(name='HARMONIC_L27',calibration='original 500 fixed trajectories; harmonic quantiles only',
        controller='harmonic raw max probability',suppression_table='harmonic_confidence_20260918/opening.npz')],
        hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=2400,
        research=dict(source='https://arxiv.org/html/2510.03223v2',location='Section 2 Eq 3; Appendix A.2 Eq 5 and Table 7',
        retained='raw max probability before penalty/temperature/top-p; step boundary, native curve, original vector/layer/amplitudes, L27 opening suppression',
        changed='arithmetic -> harmonic aggregation with matching calibration quantiles/tau',
        rejected='Do not transplant inverse-confidence attention steering sign, JSON prompting, extra masked forward, or claimed accuracy gains',
        temporal_order='sample under previous completed step state; consume saved raw max probability; at boundary compute harmonic/confidence variation/coefficient; write history; next token reads new state',
        scope='Exploratory exposed test benchmark; no independent confirmation or additive-synergy claim'),
        overhead=dict(extra_model_forwards=0,extra_answer_probes=0,per_token='reciprocal plus boundary replacement device kernels; isolated timing not measured'))
    save(out/'plan.json',plan)
    for src,dest in [(old/'historical_compact.json','historical_compact.json'),(old/'calibration_registry.json','calibration_registry.json'),
        (HERE/'label_alignment_20260917/tables/opening.npz','opening.npz')]:
        with (out/dest).open('xb') as f:f.write(src.read_bytes())
    save(out/'cpu_checks.json',dict(tests=18,passed=True,initial_test_harness_failure='AST synthesized future import missing line numbers; fixed test harness only before successful execution',
        coverage='Actual native observer on CPU vs independent harmonic scalar oracle: boundary exclusion, empty step, previous mean, history-before-use, slot mapping, invalid prefill, end-think convention; disabled no hooks; inherited lexical tests',
        limitations='Not native GPU acceptance; no calibration answer generation; greedy calibration != sampled evaluation distribution'))
    release=read(old/'release.json');release.update(artifact_root=out.name,plan_relative_path=out.name+'/plan.json',plan_sha256=sha(out/'plan.json'))
    for n in ['harmonic_adapter.py','audit_harmonic_cpu.py','test_harmonic_cpu.py','prepare_harmonic.py']:
        release['source_sha256'][n]='pending'
    release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in release['source_sha256']}
    release['artifact_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    release['cpu_checks']=read(out/'cpu_checks.json')
    save(out/'release.json',release)
    print('READY',sha(out/'release.json'))
if __name__=='__main__':main()
