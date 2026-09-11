"""Freeze three-arm feedback screen after a compiled-off identity failure."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from mechanism_candidates import ROOT,BASE,read,save,sha,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(not out.exists(),'New immutable bundle required')
    prior=ROOT/BASE/'configs/latent_feedback_screen100_cachefix_20260912'
    plan=read(prior/'plan.json');evidence=ROOT/'.codex_work/overnight_research_20260912/feedback_engineering_complete_20260912'
    audit=read(evidence/'local_audit.json')
    require(audit['files']==45 and audit['eager_exact_outputs']==8 and audit['eager_exact_hidden_rows']==2040,'Missing eager identity evidence')
    require(audit['compiled_disabled_identical']==4 and audit['original_cache_fix_identical']==8,'Unexpected engineering evidence')
    core={name:digest for name,digest in plan['source_sha256'].items() if name.startswith('sources/') or '/eval/' in name}
    for name,digest in core.items():require(sha(ROOT/name,source=True)==digest,'Model code changed since short controls: '+name)
    shutil.copytree(prior,out)
    off=dict(plan['arms']['latent_feedback_clip'],feedback_disabled=True)
    plan['arms']={'original_dynamic':plan['arms']['original_dynamic'],'feedback_disabled':off,'latent_feedback_clip':plan['arms']['latent_feedback_clip']}
    plan.update(graph_control_comparison=True,new_answers_planned=300,run_order=list(plan['arms']),
        scope='Original vector and confidence controller; online negative-displacement clipping with BOTH original-family and same-feedback-graph off controls',
        assessment='Candidate must reduce BOTH mean thinking and total tokens >=5%, have no fewer correct answers and no more caps than EACH control. No parameter selection. Confirm only with the unchanged three-arm design on untouched200.',
        redesign_reason='Previous two-arm engineering gate FAILED and remains failed. Eager off/on-path-disabled8 outputs and2040 captured states are bitwise identical; compiled old-vs-off4/8 differ. Exact compiler fusion cause is unresolved. A new off arm isolates numerical implementation effects without changing the candidate, data, parameters, or efficacy thresholds.',
        prior_plan_sha256=sha(prior/'plan.json'),prior_engineering_audit=audit,
        prior_engineering_archive_sha256='9fe44cce7d2b011fbff885d99d4b1bd3b6c346d9cf9a1274769147b9a7144351',
        expected_gpu_minutes=[8,18],batch_timeout_seconds=2400,
        engineering_completed_receipt='/root/autodl-tmp/results/easysteer/feedback_controlled_engineering_20260912/ledger.json',
        default_server_output='/root/autodl-tmp/results/easysteer/feedback_controlled_screen100_20260912')
    plan['engineering']=dict(plan['engineering'],order=['feedback_enabled'],new_short_outputs=8,
        purpose='Complete enabled-kernel smoke only. Prior16 short controls reuse identical core source, environment, inputs and assets; no efficacy grading.',
        stop='Finite valid8 outputs, cap256, same original controller and expected layer20, clipping counters valid and at least one actual clip. Failure stops before300screen. Original compiled-off mismatch is measured by an explicit control, not relabeled as passing identity.',
        reuse_directory='/root/autodl-tmp/results/easysteer/feedback_graph_smoke8_cachefix_20260912',
        reuse_sha256={name:sha(evidence/'feedback_graph_smoke8_cachefix_20260912'/name) for name in ['original_dynamic.json','feedback_disabled.json','kernel_check.json']},
        core_sources_sha256=core,expected_gpu_minutes=[1,3],batch_timeout_seconds=600)
    plan['confirmation']['status']='untouched_conditional_three_arm_reserve'
    plan['stop_conditions'][0]='Passed synthetic compiled/CUDA math checks, exact eager-off capture identity, and valid enabled short smoke; same-graph off control mandatory.'
    plan['authorization']='User sustained8-hour1.5B research; explicitly prepared new3x100 screening design plus8 remaining short engineering outputs. Not a frozen benchmark rerun.'
    paths=set(plan['source_sha256'])|{BASE+'scripts/prepare_feedback_controlled_screen.py',BASE+'scripts/run_feedback_controlled_engineering.py',BASE+'scripts/grade_vector_batch.py'}
    plan['source_sha256']={name:sha(ROOT/name,source=True) for name in sorted(paths)}
    save(out/'plan.json',plan)
    print(json.dumps(dict(screen=100,groups=3,short_new=8,reserve=200,plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
