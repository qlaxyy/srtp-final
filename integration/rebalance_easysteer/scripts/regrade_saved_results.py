"""Apply the released ReBalance grader to saved answers; no model generation.

Run with the existing ReBalance environment, which provides latex2sympy2.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sources/ReBalance'))
from utils.parser import extract_answer, parse_ground_truth
from utils.grader import check_is_correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-name', required=True)
    parser.add_argument('--group-budget-seconds', type=int, default=1500)
    parser.add_argument('--group', choices=('baseline', 'rebalance_dynamic'),
                        help='Grade one saved group without regenerating or regrading its reference')
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    saved = json.loads(raw)
    dataset = Path(saved['protocol']['dataset'])
    with dataset.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    offset, count = saved['protocol']['offset'], saved['protocol']['limit']
    rows = rows[offset:offset + count]
    result = dict(scope='author grader on saved generations; no model inference',
                  input_sha256=hashlib.sha256(raw).hexdigest(),
                  dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  data_name=args.data_name, groups={})
    for filename in ('grader.py', 'parser.py'):
        result[filename + '_sha256'] = hashlib.sha256(
            (ROOT / 'sources/ReBalance/utils' / filename).read_bytes()).hexdigest()

    groups = (args.group,) if args.group else ('baseline', 'rebalance_dynamic')
    for group in groups:
        records = saved[group]['records']
        assert len(records) == len(rows) == count
        previous_seconds = saved[group]['summary']['group_seconds_including_grading']
        remaining = args.group_budget_seconds - previous_seconds
        if args.group_budget_seconds > 0 and remaining <= 0:
            raise TimeoutError(f'{group} has exhausted its budget')
        timer = threading.Timer(remaining, lambda: os._exit(124))
        timer.daemon = True
        if args.group_budget_seconds > 0:
            timer.start()
        started = time.monotonic()
        graded = []
        try:
            for index, (row, record) in enumerate(zip(rows, records)):
                assert row['problem'] == record['problem']
                _, gold = parse_ground_truth(row, args.data_name)
                # Same call convention as the author's check.py.
                answer = extract_answer(record['text'], args.data_name)
                correct = bool(check_is_correct(answer, gold))
                graded.append(dict(index=offset + index, author_correct=correct,
                                   math_verify_correct=record['correct']))
        finally:
            timer.cancel()
        seconds = time.monotonic() - started
        result['groups'][group] = dict(
            author_correct=sum(r['author_correct'] for r in graded),
            disagreements=[r for r in graded if r['author_correct'] != r['math_verify_correct']],
            seconds=seconds, total_group_seconds=previous_seconds + seconds,
            records=graded)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(output)
        print(group, result['groups'][group]['author_correct'], '/', count, flush=True)
    if args.group:
        return
    baseline = result['groups']['baseline']['records']
    dynamic = result['groups']['rebalance_dynamic']['records']
    result['improved_indices'] = [a['index'] for a, b in zip(baseline, dynamic)
                                  if not a['author_correct'] and b['author_correct']]
    result['degraded_indices'] = [a['index'] for a, b in zip(baseline, dynamic)
                                  if a['author_correct'] and not b['author_correct']]
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output)


if __name__ == '__main__':
    main()
