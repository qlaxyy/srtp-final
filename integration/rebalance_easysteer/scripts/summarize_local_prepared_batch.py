"""Summarize every candidate locally after GPU shutdown, including partial work."""
import argparse
from pathlib import Path
import json
import time

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from analyze_pair_uncertainty import analyze


def summarize(bundle, results):
    plan = read(bundle/'plan.json'); ledger = read(results/'ledger.json')
    require(ledger['plan_sha256'] == sha(bundle/'plan.json'), 'Results use another parent plan')
    require(ledger['status'] in ('bounded_batch_finished', 'incomplete'), 'Batch still running or unrecognized')
    for relative, digest in ledger['files_sha256'].items():
        path = (results/relative).resolve()
        require(path.is_relative_to(results.resolve()) and sha(path) == digest, 'Downloaded output differs: '+relative)
    for name in ('summarize_local_prepared_batch.py', 'analyze_pair_uncertainty.py'):
        source = BASE+'scripts/'+name
        require(sha(ROOT/source, source=True) == plan['source_sha256'][source], 'Analysis source changed: '+source)
    candidates = {}
    for name in plan['candidate_order']:
        recorded = ledger['candidates'].get(name, {})
        candidate = dict(status=recorded.get('status', 'unstarted_after_batch_failure'), stages={})
        if ledger['status'] == 'incomplete' and candidate['status'] == 'pending':
            candidate['status'] = 'incomplete_runtime_or_input_failure'
        candidates[name] = candidate
        for stage, entry in plan['candidates'][name].items():
            folder = results/name/stage; run_path = folder/'run_ledger.json'; analysis_path = folder/'analysis.json'
            outcome = dict(planned_questions=entry['count'], planned_outputs=entry['new_answers'])
            candidate['stages'][stage] = outcome
            if not run_path.exists():
                outcome['status'] = 'not_started'; continue
            run = read(run_path)
            require(run['plan_sha256'] == entry['plan_sha256'], 'Stage used another plan')
            outcome.update(status=run['status'], run_ledger_sha256=sha(run_path))
            if stage == 'engineering':
                outcome['interpretation'] = 'Short implementation check only; never an accuracy or compression result.'
                require(not analysis_path.exists(), 'Engineering outputs should not be graded')
                continue
            if not analysis_path.exists():
                outcome['interpretation'] = 'Partial or ungraded output retained; no completed paired result.'
                continue
            analysis = read(analysis_path)
            require(analysis['status'] == 'completed' and analysis['stage'] == stage and
                    analysis['candidate'] == name and analysis['plan_sha256'] == entry['plan_sha256'], 'Unverified stage analysis')
            pair = analysis['comparison']; child = read(bundle/entry['bundle']/'plan.json')
            require([r['train_index'] for r in pair['per_question']] == child['train_indices'], 'Paired result support changed')
            require(list(pair['groups']) == ['original_dynamic', name], 'Wrong arm order')
            outcome.update(status='completed_paired_result', analysis_sha256=sha(analysis_path),
                metrics={key: value for key, value in pair.items() if key != 'per_question'},
                per_question=pair['per_question'], uncertainty=analyze(analysis, repetitions=10000, seed=20260928))
            outcome['interpretation'] = ('Candidate selection on fresh training questions; not independent confirmation.'
                if stage == 'screen' else 'Fixed independent reserve; small sample and one seed do not prove a narrow accuracy non-inferiority margin.')
    children = [{key: item[key] for key in ('name', 'process_seconds', 'exit_code', 'error') if key in item}
                for item in ledger['children']]
    return dict(status='local_summary_of_all_prepared_candidates', batch_status=ledger['status'],
        parent_plan_sha256=sha(bundle/'plan.json'), batch_ledger_sha256=sha(results/'ledger.json'),
        run_commit=ledger['commit'], batch_elapsed_seconds=ledger['elapsed_seconds'],
        candidates=candidates, child_process_accounting=children,
        costs='Pure generation seconds are reported per arm; parent elapsed time includes model startup, checks and grading. Neither includes user-controlled instance boot/idle time. Hourly price was not verified.',
        confidence_intervals='Descriptive paired problem bootstrap and exact McNemar only; never changes the frozen promotion gate or repairs failed candidates.',
        model_loads=0, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable summary exists')
    started = time.perf_counter(); result = summarize(args.bundle, args.results)
    result['cpu_analysis_seconds'] = time.perf_counter()-started
    save(args.output, result)
    print(json.dumps(dict(batch_status=result['batch_status'], candidates={name:value['status']
        for name,value in result['candidates'].items()}, cpu_analysis_seconds=result['cpu_analysis_seconds'], GPU_calls=0)))


if __name__ == '__main__':
    main()
