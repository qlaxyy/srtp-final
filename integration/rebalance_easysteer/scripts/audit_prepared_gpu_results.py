"""Independently recount downloaded tokens and author labels for the fixed batch.

All subgroup breakdowns are descriptive. They cannot replace the full-sample
promotion gate or select new settings. No model or grader is loaded here.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

from mechanism_candidates import ROOT, read, save, sha, require


def audit(bundle, results):
    started = time.perf_counter(); plan = read(bundle / 'plan.json')
    ledger = read(results / 'ledger.json')
    require(ledger['status'] == 'bounded_batch_finished', 'Incomplete batch')
    require(ledger['plan_sha256'] == sha(bundle / 'plan.json'), 'Parent changed')
    for name, digest in ledger['files_sha256'].items():
        path = (results / name).resolve()
        require(path.is_relative_to(results.resolve()) and sha(path) == digest,
                'Downloaded artifact changed: ' + name)
    candidates = {}; formal_outputs = formal_tokens = engineering_outputs = 0
    generation_seconds = 0.0
    for name in plan['candidate_order']:
        entry = plan['candidates'][name]['screen']
        child = read(bundle / entry['bundle'] / 'plan.json')
        folder = results / name / 'screen'; analysis = read(folder / 'analysis.json')
        require(analysis['plan_sha256'] == entry['plan_sha256'], 'Wrong analysis')
        groups = []; author = []; metrics = {}
        for arm in child['run_order']:
            path = folder / (arm + '.json'); saved = read(path)
            grade = read(folder / (arm + '.author.json'))
            require(grade['input_sha256'] == sha(path), 'Grade uses another input')
            records = saved['rebalance_dynamic']['records']; labels = grade['records']
            require(len(records) == len(labels) == child['count'] == 100,
                    'Missing formal outputs')
            require([r['train_index'] for r in labels] == child['train_indices'],
                    'Author labels were reordered')
            counts = Counter()
            for i, (record, label) in enumerate(zip(records, labels, strict=True)):
                ids = record['token_ids']; thinking = (
                    ids.index(151649) if 151649 in ids else len(ids))
                require(record['dataset_index'] == i and record['tokens'] == len(ids)
                        and record['thinking_tokens'] == thinking
                        and len(ids) <= 16000, 'Raw token accounting differs')
                require(isinstance(label['correct'], bool), 'Invalid author label')
                counts.update(correct=int(label['correct']), total=len(ids),
                    thinking=thinking, capped=int(record['finish_reason'] == 'length'
                                                 or len(ids) == 16000))
            expected = analysis['comparison']['groups'][arm]
            for raw_key, reported in [('correct', 'correct'), ('total', 'sum_total_tokens'),
                                      ('thinking', 'sum_thinking_tokens'), ('capped', 'capped')]:
                require(counts[raw_key] == expected[reported], 'Reported count differs')
            seconds = saved['rebalance_dynamic']['summary']['generation_seconds']
            require(seconds == expected['generation_seconds'], 'Timing source differs')
            metrics[arm] = dict(counts, count=100, generation_seconds=seconds)
            groups.append(records); author.append(labels)
            formal_outputs += len(records); formal_tokens += counts['total']
            generation_seconds += seconds
        categories = {k: dict(count=0, total_token_delta=0, thinking_token_delta=0)
                      for k in ('both_correct', 'improved', 'degraded', 'both_wrong')}
        longer = shorter = same_length = identical = 0; new_caps = []; resolved_caps = []
        for i, (a, b, ga, gb) in enumerate(zip(*groups, *author, strict=True)):
            key = ('both_correct' if ga['correct'] and gb['correct'] else
                   'degraded' if ga['correct'] else 'improved' if gb['correct'] else
                   'both_wrong')
            total_delta = len(b['token_ids']) - len(a['token_ids'])
            categories[key]['count'] += 1
            categories[key]['total_token_delta'] += total_delta
            categories[key]['thinking_token_delta'] += b['thinking_tokens'] - a['thinking_tokens']
            longer += int(total_delta > 0); shorter += int(total_delta < 0)
            same_length += int(total_delta == 0)
            identical += int(a['token_ids'] == b['token_ids'])
            ca = a['finish_reason'] == 'length' or len(a['token_ids']) == 16000
            cb = b['finish_reason'] == 'length' or len(b['token_ids']) == 16000
            item = dict(index=i, train_index=child['train_indices'][i], total_token_delta=total_delta)
            if cb and not ca: new_caps.append(item)
            if ca and not cb: resolved_caps.append(item)
        a, b = metrics.values()
        require(sum(v['count'] for v in categories.values()) == 100
                and sum(v['total_token_delta'] for v in categories.values()) == b['total']-a['total'],
                'Subgroup decomposition lost an output')
        passed = (20*b['total'] <= 19*a['total'] and
                  20*b['thinking'] <= 19*a['thinking'] and
                  b['correct'] >= a['correct'] and b['capped'] <= a['capped'])
        require(passed == analysis['comparison']['passes_fixed_gate'], 'Gate differs')
        reserve = results / name / 'confirmation'
        require(not passed and not reserve.exists(), 'This completed batch should leave both reserves untouched')
        eng = results / name / 'engineering'; eng_ledger = read(eng / 'run_ledger.json')
        require(eng_ledger['status'] == 'engineering_passed_no_efficacy_claim', 'Engineering failed')
        for arm in child['run_order']:
            engineering_outputs += len(read(eng / (arm + '.json'))['rebalance_dynamic']['records'])
        candidates[name] = dict(groups=metrics, passes_fixed_gate=passed,
            descriptive_correctness_groups=categories, longer=longer, shorter=shorter,
            same_length=same_length, identical_token_sequences=identical,
            newly_capped=new_caps, resolved_caps=resolved_caps, independent_reserve_outputs=0,
            analysis_sha256=sha(folder / 'analysis.json'))
    return dict(status='downloaded_raw_token_and_author_label_recount_passed',
        parent_plan_sha256=sha(bundle / 'plan.json'), batch_ledger_sha256=sha(results / 'ledger.json'),
        source_sha256=sha(Path(__file__), source=True), candidates=candidates,
        formal_outputs=formal_outputs, formal_tokens=formal_tokens,
        engineering_outputs=engineering_outputs, formal_generation_seconds=generation_seconds,
        batch_elapsed_seconds=ledger['elapsed_seconds'], cpu_seconds=time.perf_counter()-started,
        limitations='Recounts saved token IDs and already completed author labels; does '
            'not rerun grading or timing. Correctness/cap subgroups are retrospective '
            'accounting only. No subset exclusion, threshold change, new generation or '
            'independent confirmation is implied.', model_loads=0, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Preserve previous audit')
    result = audit(args.bundle.resolve(), args.results.resolve()); save(args.output, result)
    print({k: v for k, v in result.items() if k != 'candidates'})


if __name__ == '__main__':
    main()
