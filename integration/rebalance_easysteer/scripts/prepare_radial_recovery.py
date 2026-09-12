"""Retain failed radial bundle, freeze precision correction with unchanged data."""
import argparse
import json
import shutil
from pathlib import Path
from mechanism_candidates import ROOT,BASE,read,save,sha,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();require(not out.exists(),'Recovery bundle exists')
    old=ROOT/BASE/'configs/radial_screen100_20260912'
    prior=read(old/'plan.json');require(sha(old/'plan.json')=='27446f6b211f3452cfab7fea2c432dfa4b3523ae950acf7575fc77315a1382cf','Original plan changed')
    failure=read(ROOT/'.codex_work/overnight_research_20260912/radial_engineering_failure_20260912/radial_engineering8_20260912/ledger.json')
    require(failure['status']=='incomplete' and not failure['arms'],'Unexpected prior generation')
    diagnosis=ROOT/'.codex_work/overnight_research_20260912/radial_numeric_diagnostic_20260912'
    local=read(diagnosis/'local_analysis.json');receipt=read(diagnosis/'receipt.json')
    require(local['violation_mask_matches'] and local['measured_violations']==43,'Unexplained failure')
    require(sha(diagnosis/'actual_arrays.npz')==receipt['arrays_sha256'],'Diagnostic arrays changed')
    unit=json.loads(a.unit_log.read_text(encoding='utf-8').splitlines()[-1])
    require(unit['status']=='CPU_tests_passed' and unit['count']==22 and unit['GPU_calls']==0,'Corrected CPU checks incomplete')
    shutil.copytree(old,out)
    shutil.copyfile(a.unit_log,out/'cpu_tests.log')
    prior['cpu_unit_log_sha256']=sha(out/'cpu_tests.log',source=True)
    prior['precision_correction']=dict(prior_plan_sha256=sha(old/'plan.json'),
        prior_failed_commit=failure['commit'],prior_failed_engineering_status=failure['status'],
        diagnostic_receipt_sha256=sha(diagnosis/'receipt.json'),diagnostic_arrays_sha256=receipt['arrays_sha256'],
        local_analysis_sha256=sha(diagnosis/'local_analysis.json'),unit=unit,
        correction='Complete hidden+residual and coefficient*direction used by radial geometry are explicitly FP32. Final displacement remains model dtype; matched disabled and original controls remain mandatory. No compiler option or tolerance changed.',
        preservation='Original failed bundle remains immutable. Same8 engineering,100 screen and200 reserve inputs; no model generation occurred previously. Same kernel tolerances, cap, parameters, vector, layer, controls and efficacy gate.')
    prior['engineering_completed_receipt']='/root/autodl-tmp/results/easysteer/radial_fp32_engineering8_20260912/ledger.json'
    prior['default_server_output']='/root/autodl-tmp/results/easysteer/radial_fp32_screen100_20260912'
    names=set(prior['source_sha256'])|{BASE+'scripts/prepare_radial_recovery.py'}
    prior['source_sha256']={n:sha(ROOT/n,source=True) for n in sorted(names)}
    for name in ['screen100.jsonl','smoke8.jsonl','confirmation200.jsonl']:
        require(sha(out/name)==sha(old/name),'Recovery changed dataset')
    save(out/'plan.json',prior)
    from run_vector_batch import validate_bundle
    validate_bundle(out)
    print(dict(status='recovery_prepared_same_data_and_gates',plan_sha256=sha(out/'plan.json')))


if __name__=='__main__':main()
