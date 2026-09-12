"""CPU coverage and causal-state checks for control inside long paragraphs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from types import SimpleNamespace

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from audit_control_alignment import load_controller, tensor
from audit_sampled_probabilities import step_statistics, params_from_plan


def events(ids, probabilities, boundaries, end_id, interval):
    """Observe arrived tokens; virtual points never reset paragraph statistics."""
    require(isinstance(interval, int) and not isinstance(interval, bool) and interval > 0, 'Invalid interval')
    if probabilities is not None:
        require(len(ids) == len(probabilities), 'Probability length mismatch')
    total = 0; count = 0; sums = np.float32(0); previous = None
    natural = []; virtual = []; after = 0; max_gap = 0
    for position, token in enumerate(ids):
        if token == end_id: break
        total += 1
        if token in boundaries:
            if not count: continue
            if probabilities is not None:
                mean = sums / np.float32(count)
                variance = np.float32(0) if previous is None else (mean-previous)**np.float32(2)/np.float32(4)
                natural.append((position, count, float(mean), float(variance)))
                previous = mean
            count = 0; sums = np.float32(0)
            continue
        count += 1; max_gap = max(count, max_gap); after += count > interval
        if probabilities is not None:
            sums += np.float32(probabilities[position])
        if count % interval == 0:
            if probabilities is None:
                virtual.append((position, count, None, None))
            else:
                mean = sums / np.float32(count)
                variance = np.float32(0) if previous is None else (mean-previous)**np.float32(2)/np.float32(4)
                virtual.append((position, count, float(mean), float(variance)))
    return dict(thinking_tokens=total, maximum_content_gap=max_gap,
                tokens_after_interval=after, natural=natural, virtual=virtual)


def checks():
    ids = [1, 2, 3, 4, 5, 9, 9, 1, 2, 8, 1, 2, 9]
    p = np.linspace(.2, 1., len(ids), dtype=np.float32)
    seen = events(ids, p, {9}, 8, 2)
    require([r[0] for r in seen['virtual']] == [1, 3, 8], 'Virtual pulse alignment failed')
    require([r[0] for r in seen['natural']] == [5], 'Natural boundary changed')
    require(seen['thinking_tokens'] == 9 and seen['tokens_after_interval'] == 3, 'End/gap counting failed')
    expected = step_statistics(ids, np.column_stack([p, p]), {9}, 8)
    np.testing.assert_array_equal(np.array(seen['natural']), np.array([(i, n, m[0], v[0]) for i, n, m, v in expected]))
    for length in range(1, len(ids)+1):
        prefix = events(ids[:length], p[:length], {9}, 8, 2)
        for kind in ['virtual', 'natural']:
            require(prefix[kind] == [r for r in seen[kind] if r[0] < length], 'Future suffix changed past event')
    require(events([9, 9, 8], [1., 1., 1.], {9}, 8, 2)['virtual'] == [], 'Empty boundary creates pulse')
    return 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); require(not a.output.exists(), 'Immutable output exists')
    started = time.perf_counter(); nchecks = checks()
    plan_path = ROOT/BASE/'configs/interval_control_20260912.json'; plan = read(plan_path)
    old_plan_path = ROOT/BASE/'configs/seal_comparison130_20260912/plan.json'; old_plan = read(old_plan_path)
    params = params_from_plan(old_plan)
    boundaries = set(params.boundary_token_ids); interval = plan['intervention']['content_tokens_between_opportunities']
    dynamic_path = ROOT/plan['CPU_inputs']['dynamic_development']; dynamic = read(dynamic_path)
    ledger = read(dynamic_path.parent/'ledger.json')
    require(sha(dynamic_path) == ledger['files_sha256']['math_train_original_dynamic'], 'Saved dynamic answers changed')
    require(dynamic['status'] == 'completed' and dynamic['arm'] == 'original_dynamic' and len(dynamic['records']) == 100, 'Wrong dynamic group')
    require(dynamic['plan_sha256'] == sha(old_plan_path, source=True), 'Dynamic plan changed')
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    per_question = []
    for i, (row, train_index) in enumerate(zip(dynamic['records'], old_plan['math_train_indices'], strict=True)):
        require(row['problem'] == train[train_index]['problem'] and row['gold'] == train[train_index]['answer'], 'Question binding changed')
        require(row['dataset_index'] == i and len(row['token_ids']) == row['tokens'] <= 16000, 'Count mismatch')
        require((row['finish_reason'] == 'length') == (row['tokens'] == 16000), 'Cap mismatch')
        result = events(row['token_ids'], None, boundaries, params.think_end_token_id, interval)
        require(result['thinking_tokens'] == row['thinking_tokens'], 'Thinking count mismatch')
        per_question.append(dict(index=i, train_index=train_index, capped=row['finish_reason'] == 'length',
            **{k: result[k] for k in ['thinking_tokens', 'maximum_content_gap', 'tokens_after_interval']},
            virtual_points=len(result['virtual']), virtual_token_offsets=[r[0] for r in result['virtual']]))
    digest = hashlib.sha256(); calibration = []; virtual_rows = []; natural_checked = 0
    archive = ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive) as tar:
        for q, line in enumerate(tar.extractfile('generations.jsonl')):
            digest.update(line); row = json.loads(line); ids = row['token_ids']; p = np.exp(np.array(row['logprobs']))
            result = events(ids, p, boundaries, params.think_end_token_id, interval)
            expected = step_statistics(ids, np.column_stack([p, p]), boundaries, params.think_end_token_id)
            np.testing.assert_array_equal(np.array(result['natural']), np.array([(i, n, m[0], v[0]) for i, n, m, v in expected]))
            natural_checked += len(expected)
            virtual_rows.extend((q, *r) for r in result['virtual'])
            calibration.append(dict(question=q, capped=row['finish_reason'] == 'length',
                **{k: result[k] for k in ['thinking_tokens', 'maximum_content_gap', 'tokens_after_interval']},
                virtual_points=len(result['virtual'])))
    require(q == 499 and digest.hexdigest() == '4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3', 'Calibration changed')
    data = np.array(virtual_rows); runtime = load_controller()
    coefficients = np.asarray(runtime['compute_rebalance_coefficient'](tensor(data[:, 3], dtype=np.float32), tensor(data[:, 4], dtype=np.float32), params))
    require(np.isfinite(coefficients).all(), 'Nonfinite virtual coefficient')
    def summarize(rows):
        total = sum(r['thinking_tokens'] for r in rows); after = sum(r['tokens_after_interval'] for r in rows)
        return dict(questions=len(rows), caps=sum(r['capped'] for r in rows), thinking_tokens=total,
            virtual_points=sum(r['virtual_points'] for r in rows), affected_questions=sum(r['virtual_points'] > 0 for r in rows),
            tokens_after_interval=after, fraction_after_interval=after/total)
    dynamic_summary = summarize(per_question); gate = plan['CPU_gate']
    passed = (dynamic_summary['affected_questions'] >= gate['minimum_dynamic_questions_with_virtual_point'] and
              dynamic_summary['fraction_after_interval'] >= gate['minimum_dynamic_thinking_token_fraction_after_128_content_tokens_in_current_paragraph'])
    output = dict(status='completed_CPU_interval_coverage_not_generation',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        dynamic_answer_sha256=sha(dynamic_path), dynamic_answer_commit=dynamic['commit'], calibration_answer_sha256=digest.hexdigest(),
        source_plan_sha256=sha(old_plan_path, source=True), CPU_checks=nchecks, original_natural_boundary_statistics_exactly_checked=natural_checked,
        interval=interval, dynamic=dynamic_summary, calibration=summarize(calibration),
        calibration_virtual_coefficient_quantiles=np.quantile(coefficients, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        calibration_virtual_positive=int((coefficients > 0).sum()), calibration_virtual_abs_ge_point1=int((abs(coefficients) >= .1).sum()),
        dynamic_per_question=per_question, calibration_per_question=calibration, passes_fixed_gate=bool(passed),
        decision='implement_and_prepare_fresh_GPU_screen' if passed else 'stop_no_GPU_no_interval_search',
        cpu_seconds=time.perf_counter()-started, model_loads=0, GPU_calls=0, new_answers=0, interpretation=gate['interpretation'])
    save(a.output, output)
    print({k: output[k] for k in ['status', 'dynamic', 'calibration', 'original_natural_boundary_statistics_exactly_checked', 'passes_fixed_gate', 'decision', 'cpu_seconds']})


if __name__ == '__main__': main()
