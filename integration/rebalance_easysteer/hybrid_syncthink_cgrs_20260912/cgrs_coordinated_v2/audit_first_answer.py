"""Prefix-only first-box detector; conservative offline answer comparison.

No model calls, stopping simulation, or new accuracy claims.
"""
import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from audit_reasoning_tradeoffs import boxes
from engineering import save, sha


def canonical(value):
    return re.sub(r'\s+', '', str(value))


def rational(value):
    value = canonical(value)
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value):
        return Fraction(value)
    match = re.fullmatch(r'([+-]?)\\(?:d?frac|tfrac)\{([+-]?\d+)\}\{([+-]?\d+)\}', value)
    if match and int(match[3]):
        return (-1 if match[1] == '-' else 1) * Fraction(int(match[2]), int(match[3]))
    return None


def compare(a, b):
    if canonical(a) == canonical(b):
        return 'equal_exact'
    x, y = rational(a), rational(b)
    if x is not None and y is not None:
        return 'equal_rational' if x == y else 'different_rational'
    return 'unknown'


def first_answer(text):
    thinking = text.split('</think>', 1)[0]
    found = boxes(thinking)
    return (found[0] if found else None), thinking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.raw / 'manifest.json').read_text())
    groups, rows, inputs = {}, [], {}
    for seed in (142, 242):
        for arm in ('R', 'RC14', 'RChistory'):
            folder = args.raw / 'results' / f's{seed}_{arm}'
            paths = [folder / 'result.json', folder / 'stopped_batch_author_labels.jsonl']
            for path in paths:
                assert sha(path) == manifest[path.relative_to(args.raw).as_posix()]
                inputs[str(path)] = sha(path)
            result = json.loads(paths[0].read_text())
            labels = {r['dataset_index']: r for r in map(json.loads, paths[1].read_text().splitlines())}
            assert result['status'] == 'complete' and len(result['records']) == 100
            batch = []
            for record in result['records']:
                label = labels[record['dataset_index']]
                assert label['text_sha256'] == hashlib.sha256(record['text'].encode()).hexdigest()
                assert label['problem_sha256'] == record['problem_sha256']
                first, thinking = first_answer(record['text'])
                if first is None:
                    continue
                relation = compare(first['value'], record['answer'])
                item = dict(seed=seed, arm=arm, train_index=record['train_index'],
                            problem_sha256=record['problem_sha256'], first_box=first,
                            gold=record['answer'], first_to_gold=relation,
                            final_correct=label['correct'], capped=record['finish_reason']=='length',
                            thinking_chars=len(thinking), remaining_chars=len(thinking)-first['end'],
                            context=thinking[max(0,first['start']-300):first['end']+300])
                batch.append(item)
            rows.extend(batch)
            groups[f's{seed}_{arm}'] = dict(n=100, covered=len(batch),
                first_matches_gold=sum(r['first_to_gold'].startswith('equal') for r in batch),
                first_matches_gold_final_wrong=sum(r['first_to_gold'].startswith('equal') and not r['final_correct'] for r in batch),
                first_different_rational_final_correct=sum(r['first_to_gold']=='different_rational' and r['final_correct'] for r in batch),
                unknown_comparison=sum(r['first_to_gold']=='unknown' for r in batch),
                total_thinking_chars=sum(len(r['text'].split('</think>',1)[0]) for r in result['records']),
                remaining_chars_sum=sum(r['remaining_chars'] for r in batch),
                remaining_chars_sorted=sorted(r['remaining_chars'] for r in batch))
    save(args.output/'rows.json', rows)
    save(args.output/'audit.json', dict(groups=groups,input_sha256=inputs,source_sha256=sha(Path(__file__),True),
        rule='First complete boxed expression in thinking; no threshold, gold or future text in detector.',
        limitations=['600 saved trajectories from 100 exposed questions, not 600 independent questions.',
                     'Box may denote an intermediate quantity, not proposed final answer.',
                     'Only exact whitespace-normalized equality or simple rational equality/difference is assessed; other comparisons unknown.',
                     'Remaining characters are not saved tokens; no counterfactual generation or stopping accuracy measured.',
                     'Gold and final correctness are offline diagnostic annotations only.'],gpu_ready=False))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='remaining_chars_sorted'} for k,v in groups.items()},indent=2))


if __name__ == '__main__':
    main()
