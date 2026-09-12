"""Compare two confidence definitions on the same saved, unsteered prefixes."""
import argparse
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from mechanism_candidates import ROOT, read, save, sha, require
from audit_control_alignment import load_controller, tensor, scalar_reference


def params_from_plan(plan):
    """Curve-only plans obtain exact markers from the verified frozen tokenizer."""
    path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    require(sha(path) == plan['model_files_sha256']['tokenizer.json'], 'Tokenizer changed')
    tokenizer = read(path); vocab = dict(tokenizer['model']['vocab'])
    vocab.update({t['content']: t['id'] for t in tokenizer['added_tokens']})
    markers = dict(boundary_token_ids=sorted(v for k, v in vocab.items() if 'ĊĊ' in k),
                   think_start_token_id=vocab['<think>'], think_end_token_id=vocab['</think>'])
    require(markers['boundary_token_ids'] and markers['think_start_token_id'] == 151648 and
            markers['think_end_token_id'] == 151649, 'Unexpected frozen markers')
    values = dict(plan['dynamic_parameters'])
    for key, value in markers.items():
        require(key not in values or values[key] == value, 'Conflicting token markers')
        values[key] = value
    return SimpleNamespace(**values)


def step_statistics(ids, probabilities, boundaries, end_id):
    """Match sequential FP32 accumulation and last nonempty-step variance."""
    sums = np.zeros(2, dtype=np.float32); previous = None; count = 0; records = []
    for index, (token, p) in enumerate(zip(ids, probabilities, strict=True)):
        if token == end_id: break
        if token not in boundaries:
            sums += np.asarray(p, dtype=np.float32); count += 1; continue
        if not count: continue
        means = sums/np.float32(count)
        variance = np.zeros(2, dtype=np.float32) if previous is None else (means-previous)**2/np.float32(4)
        records.append((index, count, means.copy(), variance.copy()))
        previous = means.copy(); sums.fill(0); count = 0
    return records


def checks():
    ids = [1, 2, 9, 9, 3, 9, 8, 4, 9]
    p = np.array([[.8, .4], [.6, .6], [1, 1], [1, 1], [.5, .25], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
    steps = step_statistics(ids, p, {9}, 8)
    require([s[0] for s in steps] == [2, 5] and [s[1] for s in steps] == [2, 1], 'Boundary/end semantics failed')
    np.testing.assert_allclose(steps[0][2], [.7, .5], atol=1e-7)
    np.testing.assert_array_equal(steps[0][3], [0, 0])
    np.testing.assert_allclose(steps[1][3], [.01, .015625], atol=1e-7)
    require(step_statistics([9, 8], np.ones((2, 2)), {9}, 8) == [], 'Empty-step handling failed')
    # Raw selected/max definitions agree for modal choices and ignore sampling temperature.
    logits = np.log(np.array([[.6, .4], [.2, .8]])); probabilities = np.exp(logits)
    selected = probabilities[np.arange(2), [1, 1]]
    np.testing.assert_allclose(selected, [.4, .8], atol=1e-15)
    require(selected[0] < probabilities[0].max() and selected[1] == probabilities[1].max(), 'Selected probability identity failed')
    return 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    a = parser.parse_args(); nchecks = checks()
    if a.self_test: print(dict(CPU_checks=nchecks, GPU_calls=0, model_loads=0)); return
    require(a.bundle and a.replay and a.output and not a.output.exists(), 'Fresh analysis output required')
    started = time.perf_counter(); plan = read(a.bundle/'plan.json'); ledger = read(a.replay/'ledger.json')
    require(ledger['status'] == 'completed' and ledger['completed_questions'] == 100, 'Incomplete replay')
    require(ledger['plan_sha256'] == sha(a.bundle/'plan.json'), 'Replay plan changed')
    require(sha(a.bundle/'questions.json') == plan['question_mapping_sha256'], 'Question mapping changed')
    params = params_from_plan(plan); boundary = set(params.boundary_token_ids)
    data = []; all_max = []; all_selected = []; nonmodal = []; per_question = []
    for q in read(a.bundle/'questions.json'):
        file = f"q{q['index']:03d}.npz"; path = a.replay/file; require(sha(path) == ledger['files_sha256'][file], 'Probability file changed')
        arrays = np.load(path); ids, maximum, selected, argmax = [arrays[k] for k in ['token_ids', 'maximum', 'selected', 'argmax']]
        require(hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest() == q['thinking_token_ids_sha256'], 'Replayed token prefix changed')
        require(len(ids) == q['thinking_tokens'] and np.isfinite(maximum).all() and np.isfinite(selected).all(), 'Invalid population')
        require(np.all((selected >= 0)&(selected <= maximum)&(maximum <= 1)), 'Probability order failed')
        require(np.array_equal(selected[ids == argmax], maximum[ids == argmax]), 'Greedy equality failed')
        steps = step_statistics(ids, np.column_stack([maximum, selected]), boundary, params.think_end_token_id)
        for position, count, mean, variance in steps:
            data.append((q['index'], position, count, mean[0], mean[1], variance[0], variance[1]))
        all_max.append(maximum); all_selected.append(selected); nonmodal.append(ids != argmax)
        per_question.append(dict(index=q['index'], train_index=q['train_index'], nonempty_boundaries=len(steps),
            thinking_tokens=len(ids), nonmodal_token_fraction=float((ids != argmax).mean()),
            mean_modal_minus_selected_probability=float((maximum-selected).mean())))
    data = np.array(data); require(len(data) > 0, 'No boundaries')
    runtime = load_controller(); compute = runtime['compute_rebalance_coefficient']
    modal_c = np.asarray(compute(tensor(data[:, 3], dtype=np.float32), tensor(data[:, 5], dtype=np.float32), params))
    selected_c = np.asarray(compute(tensor(data[:, 4], dtype=np.float32), tensor(data[:, 6], dtype=np.float32), params))
    # Independent FP64 scalar decomposition tolerates only single-precision arithmetic differences.
    constants = runtime['_curve_constants'](params.q25c, params.q75c, params.low_val_1, 'cpu', np.dtype('float64'), params.curve_tau)
    error = max(abs(scalar_reference(float(row[3]), float(row[5]), params, constants)['coefficient']-float(c)) for row, c in zip(data, modal_c, strict=True))
    require(error < .001, 'Controller implementation mismatch')
    delta = selected_c-modal_c; gate = plan['CPU_opportunity_gate']; affected = abs(delta) >= gate['minimum_absolute_coefficient_difference']
    for q in per_question:
        mask = data[:, 0] == q['index']; q.update(affected_boundaries=int((mask & affected).sum()),
            affected_fraction=float(affected[mask].mean()) if mask.any() else None)
    affected_questions = int(len(np.unique(data[affected, 0])))
    passed = affected.mean() >= gate['minimum_fraction_of_nonempty_step_boundaries'] and affected_questions >= gate['minimum_affected_questions']
    result = dict(status='completed_offline_probability_opportunity_not_generation', plan_sha256=sha(a.bundle/'plan.json'),
        script_sha256=sha(Path(__file__), source=True), replay_ledger_sha256=sha(a.replay/'ledger.json'), CPU_checks=nchecks,
        source_answer_sha256=plan['source_answer_sha256'], replay_commit=ledger['commit'],
        questions=100, capped_answers_included=8, thinking_tokens=sum(map(len, all_max)), nonempty_step_boundaries=len(data),
        nonmodal_token_fraction=float(np.concatenate(nonmodal).mean()),
        pooled_mean_modal_probability=float(np.concatenate(all_max).mean()), pooled_mean_selected_probability=float(np.concatenate(all_selected).mean()),
        affected_boundaries=int(affected.sum()), affected_fraction=float(affected.mean()), affected_questions=affected_questions,
        coefficient_delta_quantiles=np.quantile(delta, [0, .1, .25, .5, .75, .9, .99, 1]).tolist(),
        sign_changed_boundaries=int(np.count_nonzero(np.sign(modal_c) != np.sign(selected_c))),
        scalar_controller_max_error=error, per_question=per_question, passes_fixed_gate=bool(passed),
        decision='prepare_fresh_generation_screen' if passed else 'stop_no_generation_no_retuning',
        cpu_seconds=time.perf_counter()-started, new_answers=0, GPU_calls=0, limitations=plan['limitations'])
    save(a.output, result)
    print({k: result[k] for k in ['status', 'nonmodal_token_fraction', 'nonempty_step_boundaries', 'affected_fraction', 'affected_questions', 'passes_fixed_gate', 'decision', 'cpu_seconds']})


if __name__ == '__main__': main()
