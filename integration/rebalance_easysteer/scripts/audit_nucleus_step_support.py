"""Lift a sufficient token-level nucleus condition to saved complete steps.

This conditions on OLD GREEDY prefixes. It is not stochastic replay, an online
opportunity estimate, or a change to either prepared candidate or its gates.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, read, save, sha, require


def audit():
    started = time.perf_counter()
    work = ROOT / '.codex_work/overnight_research_20260912'
    bound_path = work / 'sampled_nucleus_bound_CPU.json'
    bound = read(bound_path)
    require(sha(bound_path) ==
            '0395d6261a98241436474b43c2abd2a087359b3a7f7503f886b8582532c71d0d',
            'Original bound audit changed')
    step_path = ROOT / '.codex_work/question_balanced_20260911/original_selected_layer/steps.json'
    require(sha(step_path) ==
            'fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93',
            'Original complete-step support changed')
    steps = read(step_path); by_question = defaultdict(list)
    require(len(steps) == 84008, 'Wrong complete-step support')
    for step in steps:
        require(step['start'] < step['stop'], 'Empty step needs another definition')
        by_question[step['question']].append(step)
    require(set(by_question) == set(range(500)), 'Wrong question support')
    archive = ROOT / '.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    threshold = bound['sufficient_raw_maximum'] + bound['numeric_margin']
    rows = []; digest = hashlib.sha256(); checked = 0
    with tarfile.open(archive) as tar:
        for question, line in enumerate(tar.extractfile('generations.jsonl')):
            digest.update(line); raw = json.loads(line)
            maximum = np.exp(np.array(raw['logprobs'], dtype=np.float64))
            certified = maximum > threshold
            prefix_bad = np.r_[0, np.cumsum(~certified)]
            previous_mean = None; previous_certified = True
            current_count = pair_count = content_count = certified_content = 0
            for step in by_question[question]:
                begin, end = step['start'], step['stop']
                probabilities = maximum[begin:end]
                require(len(probabilities) == end-begin, 'Step exceeds saved tokens')
                mean = float(probabilities.mean())
                variance = 0.0 if previous_mean is None else (mean-previous_mean)**2/4
                require(abs(mean-step['confidence']) < 1e-12 and
                        abs(variance-step['variance']) < 1e-12,
                        'Original complete-step statistics do not reconstruct')
                current_certified = bool(prefix_bad[end] == prefix_bad[begin])
                require(current_certified == bool(certified[begin:end].all()),
                        'Independent interval count differs')
                current_count += int(current_certified)
                pair_count += int(current_certified and previous_certified)
                content_count += end-begin
                certified_content += int(certified[begin:end].sum())
                previous_mean = mean; previous_certified = current_certified; checked += 1
            rows.append(dict(question=question, train_index=raw['train_index'],
                complete_steps=len(by_question[question]),
                certified_same_step_mean=current_count,
                certified_same_mean_and_variance=pair_count,
                complete_step_content_tokens=content_count,
                certified_content_tokens=certified_content,
                capped=raw['finish_reason'] == 'length'))
    require(digest.hexdigest() == bound['generations_sha256'], 'Saved answers changed')
    require(len(rows) == 500 and checked == 84008, 'Incomplete audit')
    totals = {key: sum(r[key] for r in rows) for key in (
        'complete_steps', 'certified_same_step_mean',
        'certified_same_mean_and_variance', 'complete_step_content_tokens',
        'certified_content_tokens')}
    return dict(status='CPU_conditional_step_sufficient_bound_not_online_opportunity',
        totals=totals, per_question=rows, original_step_statistics_reconstructed=checked,
        bound_audit_sha256=sha(bound_path), steps_sha256=sha(step_path),
        generations_sha256=digest.hexdigest(), source_sha256=sha(Path(__file__), source=True),
        interpretation='On exactly these saved greedy prefixes, each certified step '
            'contains only tokens whose raw modal probability guarantees a singleton '
            'nucleus at T0.7/top_p0.95, without other processors. Its mean is unchanged '
            'under either confidence definition. If the previous complete step is '
            'also certified, the two-step variance is unchanged; the first variance '
            'is zero. This sufficient condition says nothing about changes at '
            'uncertified steps, nor prefixes reached by stochastic generation.',
        limits='Original calibration development data, including caps. Incomplete '
            'tails excluded because this audit uses the original84008complete steps. '
            'No conclusion about steering execution at each stored state, generation '
            'benefits, or real-kernel numerical identity. Prepared batch unchanged.',
        cpu_seconds=time.perf_counter()-started, GPU_calls=0, model_loads=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Preserve prior receipt')
    result = audit(); save(args.output, result)
    print({k: result[k] for k in ('status', 'totals', 'cpu_seconds', 'GPU_calls')})


if __name__ == '__main__':
    main()
