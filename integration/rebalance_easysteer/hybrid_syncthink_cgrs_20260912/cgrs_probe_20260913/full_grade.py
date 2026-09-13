"""Grade fixed full RC results and compare with hash-checked historical U/R."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time

from run import ROOT, read, save, sha
from grade import paired, accuracy_interval


def utilization(folder, result):
    offset = read(folder/'runtime_identity.json')['timezone_offset_seconds']
    values = []
    for row in csv.reader((folder/'gpu_utilization.csv').read_text().splitlines()):
        if len(row) != 5:
            continue
        stamp = datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(
            tzinfo=timezone(timedelta(seconds=offset))).timestamp()
        if result['started_epoch'] <= stamp <= result['ended_epoch']:
            values.append(float(row[1]))
    if not values:
        return dict(samples=0)
    ordered = sorted(values)
    return dict(samples=len(values), mean_percent=sum(values)/len(values),
                median_percent=ordered[len(values)//2], min_percent=min(values), max_percent=max(values),
                fraction_at_least_90=sum(x>=90 for x in values)/len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Reuse validated saved author labels; never regenerate')
    args = p.parse_args()
    folder = args.output
    plan, history = read(folder/'resolved_plan.json'), read(folder/'historical_reference.json')
    assert read(folder/'batch_status.json')['status'] == 'complete'
    sys.path.insert(0, str(ROOT/'sources/ReBalance'))
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    started, summary, reused = time.monotonic(), {}, {}
    keys = ('thinking_tokens', 'tokens', 'all_branch_output_tokens', 'budget_tokens_including_probe_prompt')
    for role, sub in plan['datasets'].items():
        data = read(folder/role/'RC/result.json')
        records = data['records']
        assert data['status'] == 'complete' and len(records) == len(sub['rows'])
        graded = []
        partial = folder/role/'RC/author_partial.jsonl'
        previous = [json.loads(line) for line in partial.read_text(encoding='utf8').splitlines()] if args.resume and partial.exists() else []
        assert len(previous) <= len(records)
        reused[role] = dict(labels=len(previous), partial_sha256=sha(partial) if partial.exists() else None)
        with partial.open('a' if args.resume else 'x', encoding='utf8') as stream:
            for i, (row, rec) in enumerate(zip(sub['rows'], records)):
                assert row['problem_sha256'] == rec['problem_sha256']
                if i < len(previous):
                    item = previous[i]
                    assert item['problem_sha256'] == row['problem_sha256']
                    assert item['dataset_index'] == row['train_index'] and type(item['correct']) is bool
                    graded.append(item)
                    continue
                _, gold = parse_ground_truth(row, sub['dataset'])
                answer = extract_answer(rec['text'], sub['dataset'])
                item = dict(problem_sha256=row['problem_sha256'], dataset_index=row['train_index'],
                            correct=bool(check_is_correct(answer, gold)))
                graded.append(item)
                stream.write(json.dumps(item)+'\n')
                stream.flush()
        comparisons = {}
        for arm, reference in history[role]['groups'].items():
            ref = reference['records']
            assert [r['problem_sha256'] for r in ref] == [r['problem_sha256'] for r in graded]
            wr = [r['dataset_index'] for r,c in zip(ref,graded) if not r['correct'] and c['correct']]
            rw = [r['dataset_index'] for r,c in zip(ref,graded) if r['correct'] and not c['correct']]
            ci = accuracy_interval(ref, graded)
            comparisons[arm] = dict(accuracy_delta_pp=100*(len(wr)-len(rw))/len(ref),
                accuracy_delta_pp_ci95=ci, wrong_to_right=wr, right_to_wrong=rw,
                point_loss_within_2pp=100*(len(wr)-len(rw))/len(ref)>=-2,
                ci95_establishes_2pp_noninferiority=ci[0]>=-2,
                lengths={key: paired(ref,records,key) for key in keys})
        summary[role] = dict(n=len(records), correct=sum(r['correct'] for r in graded),
            accuracy_percent=100*sum(r['correct'] for r in graded)/len(graded),
            means={key:sum(r[key] for r in records)/len(records) for key in keys},
            capped=sum(r['finish_reason']=='length' for r in records),
            generation_seconds=data['generation_seconds'], gpu_utilization=utilization(folder,data),
            probe_count=sum(r['policy']['probes'] for r in records),
            cost={key:data[key] for key in ('probe_output_tokens','probe_prompt_tokens',
                'probe_wall_seconds','callback_host_seconds','extra_forward_counts','clone_bytes',
                'probe_prefill_tokens','checkpoint_io_seconds')},
            comparisons=comparisons, grades=graded, result_sha256=sha(folder/role/'RC/result.json'))
    save(folder/'analysis.json', dict(status='full_evaluation_complete', datasets=summary,
        grade_seconds=time.monotonic()-started,
        resumed=args.resume, reused_grades=reused,
        grade_timing_note='This invocation only; previous interrupted CPU grading is additional unmeasured cost' if args.resume else 'Single complete grading invocation',
        analysis_implementation_sha256=sha(Path(__file__)),
        grader_sha256={name:sha(ROOT/'sources/ReBalance/utils'/name) for name in ('parser.py','grader.py')},
        limitations=['Historical U/R use asynchronous scheduling; RC uses synchronous KV-clone probes.',
            'No full C-only control: this does not establish factorial synergy or superiority over both components.',
            'Single seed; paired bootstrap resamples questions, not batches or repeated runs.',
            'Previously exposed test sets are final descriptive evaluation, not fresh independent confirmation.',
            'Probe wall and callback times overlap generation time and must not be added to it.']))


if __name__ == '__main__':
    main()
