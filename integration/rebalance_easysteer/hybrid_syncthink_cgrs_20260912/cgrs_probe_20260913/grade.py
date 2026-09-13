"""Author grading and paired screening summaries; existing grading env only."""
import argparse
import random
import sys
import time

from run import ROOT, read, save, sha


def paired(reference, candidate, key):
    a, b = [r[key] for r in reference], [r[key] for r in candidate]
    rng = random.Random(20260913)
    values = []
    for _ in range(10000):
        indices = [rng.randrange(len(a)) for _ in a]
        denominator = sum(a[i] for i in indices)
        values.append((sum(b[i] for i in indices) / denominator - 1) * 100
                      if denominator else 0.)
    values.sort()
    return dict(percent_change=(sum(b) / sum(a) - 1) * 100 if sum(a) else 0.,
                bootstrap_ci95=[values[249], values[9749]], repetitions=10000)


def accuracy_interval(reference, candidate):
    differences = [int(c['correct']) - int(r['correct'])
                   for r, c in zip(reference, candidate)]
    rng = random.Random(20260913)
    values = sorted(100*sum(rng.choice(differences) for _ in differences)/len(differences)
                    for _ in range(10000))
    return [values[249], values[9749]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from pathlib import Path
    folder = Path(args.output)
    plan = read(folder/'resolved_plan.json')
    if read(folder/'batch_status.json')['status'] != 'complete':
        raise RuntimeError('Do not grade a partial run as complete')
    if plan['phase'] != 'screen':
        raise ValueError('Engineering answers are not efficacy results')
    sys.path.insert(0, str(ROOT/'sources/ReBalance'))
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    started, results, grades = time.monotonic(), {}, {}
    for arm in ('R', 'C', 'RC'):
        data = read(folder/arm/'result.json')
        if data['status'] != 'complete' or len(data['records']) != 64:
            raise ValueError('Incomplete fixed screening arm')
        records = data['records']
        graded = []
        with (folder/arm/'author_partial.jsonl').open('x', encoding='utf8') as stream:
            import json
            for row, rec in zip(plan['rows'], records):
                if row['problem_sha256'] != rec['problem_sha256']:
                    raise ValueError('Pair identity mismatch')
                _, gold = parse_ground_truth(row, 'math')
                answer = extract_answer(rec['text'], 'math')
                item = dict(problem_sha256=row['problem_sha256'],
                            train_index=row['train_index'],
                            correct=bool(check_is_correct(answer, gold)))
                graded.append(item)
                stream.write(json.dumps(item)+'\n')
                stream.flush()
        grades[arm] = graded
        results[arm] = data
    comparisons = {}
    for ref in ('R', 'C'):
        wrong_right = [r['train_index'] for r, c in zip(grades[ref], grades['RC'])
                       if not r['correct'] and c['correct']]
        right_wrong = [r['train_index'] for r, c in zip(grades[ref], grades['RC'])
                       if r['correct'] and not c['correct']]
        comparisons['RC_minus_'+ref] = dict(
            accuracy_delta_pp=100*(len(wrong_right)-len(right_wrong))/64,
            accuracy_delta_pp_ci95=accuracy_interval(grades[ref], grades['RC']),
            wrong_to_right=wrong_right, right_to_wrong=right_wrong,
            lengths={key: paired(results[ref]['records'], results['RC']['records'], key)
                     for key in ('thinking_tokens', 'tokens', 'all_branch_output_tokens',
                                 'budget_tokens_including_probe_prompt')})
    summary = dict(status='screening_complete_not_independent_confirmation',
                   arms={arm: dict(correct=sum(x['correct'] for x in grades[arm]),
                                   n=64, capped=sum(r['finish_reason']=='length'
                                                    for r in results[arm]['records']),
                                   generation_seconds=results[arm]['generation_seconds'],
                                   means={key: sum(r[key] for r in results[arm]['records'])/64
                                          for key in ('thinking_tokens', 'tokens',
                                                      'all_branch_output_tokens',
                                                      'budget_tokens_including_probe_prompt')})
                         for arm in grades},
                   comparisons=comparisons, grade_seconds=time.monotonic()-started,
                   grades=grades,
                   input_sha256={a: sha(folder/a/'result.json') for a in grades},
                   grader_sha256={f: sha(ROOT/'sources/ReBalance/utils'/f)
                                  for f in ('parser.py', 'grader.py')},
                   limitations=['No U group: no factorial synergy claim',
                                'Single seed, synchronous pilot, paired CI excludes batch variability',
                                'Point loss <=2pp does not establish noninferiority'])
    save(folder/'analysis.json', summary)


if __name__ == '__main__':
    main()
