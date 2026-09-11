"""Freeze a new two-arm positive-branch ablation and untouched confirmation."""
import argparse
import json
from pathlib import Path
import random
import shutil

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import rows, norm, TRAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--unit-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), 'Immutable bundle exists')
    cpu = read(args.cpu)
    require(cpu['passes_opportunity_gate'], 'CPU opportunity gate failed')
    unit = json.loads(args.unit_log.read_text(encoding='utf-8').splitlines()[-1])
    require(unit['status'] == 'passed' and unit['tests'] == 18 and unit['cpu_only'], 'Unit checks incomplete')
    config = ROOT/BASE/'configs'
    prior = read(config/'seal_comparison130_20260912/plan.json')
    mechanism = read(config/'positive_branch_ablation_20260912.json')
    require(sha(config/'positive_branch_ablation_20260912.json', source=True) == cpu['mechanism_plan_sha256'], 'Mechanism changed')
    exclusions = dict(prior['exclusions'], seal_screen=prior['math_train_indices'],
                      seal_engineering=prior['engineering_indices'])
    excluded = set().union(*(set(values) for values in exclusions.values()))
    train = rows(ROOT/TRAIN)
    test_sha = {name:digest for name,digest in prior['source_inputs_sha256'].items() if name != TRAIN}
    tests = {name:rows(ROOT/name) for name in test_sha}
    for name,digest in test_sha.items():
        require(sha(ROOT/name, source=True) == digest, 'Test source changed')
    seen = {norm(r['problem']) for group in tests.values() for r in group}
    seen |= {norm(train[i]['problem']) for i in excluded}
    eligible = []
    for i,row in enumerate(train):
        text = norm(row['problem'])
        if i not in excluded and text not in seen:
            eligible.append(i)
            seen.add(text)
    chosen = random.Random(20260918).sample(eligible, 308)
    engineering, screen, reserve = chosen[:8], chosen[8:108], chosen[108:]
    splits = dict(engineering=engineering, screen=screen, confirmation=reserve)
    overlap = {}
    for name,indices in splits.items():
        prompts = {norm(train[i]['problem']) for i in indices}
        require(len(prompts) == len(indices), 'Duplicate split prompts')
        for old,values in exclusions.items():
            overlap[name+'_vs_'+old] = len(set(indices)&set(values))
        for old,group in tests.items():
            overlap[name+'_vs_'+Path(old).parent.name] = len(prompts&{norm(r['problem']) for r in group})
    require(len(set(chosen)) == 308 and not any(overlap.values()), 'Split overlap')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n*.log text eol=lf\n', encoding='utf-8', newline='\n')
    for file,indices in [('smoke8.jsonl',engineering), ('screen100.jsonl',screen), ('confirmation200.jsonl',reserve)]:
        (out/file).write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices), encoding='utf-8', newline='\n')
    backup = ROOT/read(config/'overnight_research_20260912.json')['first_investigation']['inputs']['backup']
    arms = {}
    for name in ['original_dynamic', 'negative_only_dynamic']:
        folder = out/'assets'/name
        folder.mkdir(parents=True)
        for file in ['auto_vector.pt', 'fit.json']:
            shutil.copyfile(backup/file, folder/file)
        arms[name] = dict(directory='assets/'+name, vector_sha256=sha(folder/'auto_vector.pt'), fit_sha256=sha(folder/'fit.json'))
    arms['negative_only_dynamic']['negative_only'] = True
    shutil.copyfile(args.unit_log, out/'cpu_tests.log')
    save(out/'cpu_opportunities.json', cpu)
    paths = set(prior['source_sha256'])
    paths.update(BASE+'scripts/'+name for name in ['prepare_positive_branch_screen.py', 'audit_positive_branch.py'])
    plan = dict(status='prepared_not_run', stage='screen', positive_branch_ablation=True,
        scope='Fresh100 MATH training causal ablation of all positive coefficients; preserve original negative branch and all calibration assets',
        count=100, new_answers_planned=200, dataset_file='screen100.jsonl', dataset_sha256=sha(out/'screen100.jsonl'),
        selection_seed=20260918, train_indices=screen, eligible_count=len(eligible), exclusions=exclusions, overlap_checks=overlap,
        confirmation=dict(count=200, train_indices=reserve, dataset_file='confirmation200.jsonl', dataset_sha256=sha(out/'confirmation200.jsonl'), status='untouched_conditional_reserve'),
        engineering=dict(count=8, train_indices=engineering, dataset_file='smoke8.jsonl', dataset_sha256=sha(out/'smoke8.jsonl'), max_tokens=1024,
            order=['original_dynamic','ablation_disabled','negative_only_dynamic'], new_short_outputs=24,
            purpose='Explicitly disabled option must match default token outputs; enabled option must cancel positive boundary updates. Longer short cap1024 allows sparse high-confidence updates; no efficacy grading.',
            stop='Any identity, source, runtime, cancellation-count, completeness or deadline failure stops before100-screen. No automatic rerun.',
            expected_gpu_minutes=[2,5], batch_timeout_seconds=900, group_timeout_seconds=300),
        engineering_completed_receipt='/root/autodl-tmp/results/easysteer/positive_branch_engineering8_20260912/ledger.json',
        train_sha256=sha(ROOT/TRAIN,source=True), test_prompt_source_sha256=test_sha,
        model=prior['model'], model_files_sha256=prior['model_files_sha256'], decoder_output_layer=20,
        dynamic_parameters=prior['dynamic_parameters'], runtime=prior['runtime'], arms=arms, run_order=list(arms),
        source_sha256={name:sha(ROOT/name,source=True) for name in sorted(paths)},
        cpu_opportunity_sha256=sha(out/'cpu_opportunities.json'), cpu_unit_log_sha256=sha(out/'cpu_tests.log',source=True), cpu_unit_tests=unit,
        hypothesis=mechanism, assessment=mechanism['proposed_generation_scope']['screen_gate'],
        expected_gpu_minutes=[6,12], batch_timeout_seconds=1500,
        default_server_output='/root/autodl-tmp/results/easysteer/positive_branch_screen100_20260912',
        stop_conditions=mechanism['proposed_generation_scope']['screen_gate'],
        limitations=mechanism['limitations']+['Counts record proposed positive updates at completed boundaries; a final capped delimiter may have no subsequent forward.',
            'The original negative branch, original raw vector and same additive graph are retained; no feedback, SEAL or first-prompt injection is combined.'])
    save(out/'plan.json', plan)
    print(json.dumps(dict(screen=100, engineering=8, reserve=200, eligible=len(eligible), plan_sha256=sha(out/'plan.json'))))


if __name__ == '__main__':
    main()
