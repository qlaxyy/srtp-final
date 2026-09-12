"""Fit one fixed empirical crossing quantile, retaining original vector bytes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from types import SimpleNamespace

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require, moments, read_vector
from audit_control_alignment import load_controller, tensor
from audit_sampled_probabilities import step_statistics, params_from_plan


def fixed_amplitude(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), 'Invalid calibration scores')
    return max(.5, float(np.quantile(values, .95, method='higher')))


def checks():
    original = np.r_[np.linspace(.2, 1., 99), 1000.]
    require(fixed_amplitude(original) == fixed_amplitude(np.r_[original[:-1], 1e9]), 'One extreme tail still determines amplitude')
    threshold = fixed_amplitude(original)
    require(np.mean(original <= threshold) >= .95 and threshold < original.max(), 'Empirical coverage identity failed')
    require(fixed_amplitude([-.2, .1, .3]) == .5, 'Moderate bound not retained')
    for invalid in [[], [np.nan], [np.inf]]:
        try: fixed_amplitude(invalid)
        except ValueError: pass
        else: raise ValueError('Malformed values accepted')
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); out = a.output.resolve(); require(not out.exists(), 'Immutable output exists')
    started = time.perf_counter(); nchecks = checks()
    plan_path = ROOT/BASE/'configs/robust_aggressive_20260912.json'; plan = read(plan_path)
    backup = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    source_plan = read(ROOT/'.codex_work/overnight_research_20260912/control_point_prepared/replay_plan.json')
    names = ['steps.json', 'fit.json', 'auto_vector.pt', 'layer_21.npy']
    source_hashes = {name: sha(backup/name) for name in names}
    for name, digest in source_hashes.items():
        require(digest == source_plan['original_files_sha256'][name], 'Original file changed: '+name)
    fit = read(backup/'fit.json'); steps = read(backup/'steps.json'); selection = read(backup/'selected_layer.json')
    c = np.array([r['confidence'] for r in steps]); question = np.array([r['question'] for r in steps])
    lexical = np.array([r['lexical_hit'] for r in steps]); old_params = fit['parameters']
    over = lexical | (c < old_params['q25c']); under = ~lexical & (c > old_params['q75c'])
    x = np.load(backup/'layer_21.npy', mmap_mode='r'); require(x.shape == (84008, 1536), 'Wrong states')
    statistics = moments(x, dict(over=over, under=under))
    mo, mu = statistics['over']['mean'], statistics['under']['mean']; direction = mo-mu
    saved_vector = read_vector(backup/'auto_vector.pt', backup/'auto_vector.pt')
    require(np.array_equal(direction.astype(np.float32), saved_vector), 'Original raw vector does not reproduce')
    w = direction/(statistics['over']['variance']+statistics['under']['variance']+1e-12)
    projection = float(w@direction); separator = float(.5*w@(mo+mu))
    scores = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 2048):
        scores[start:start+2048] = (np.asarray(x[start:start+2048], dtype=np.float64)@w-separator)/projection
    original_amplitude = float(scores[over].max())
    require(abs(original_amplitude+old_params['low_val_2']) < 1e-10, 'Original maximum does not reproduce')
    amplitude = fixed_amplitude(scores[over]); ratio = 1-amplitude/original_amplitude
    groups = {key: np.isin(question, selection[key]) for key in ['training_questions', 'validation_questions']}
    group_quantiles = {key: fixed_amplitude(scores[over & mask]) for key, mask in groups.items()}
    stability = min(group_quantiles.values())/max(group_quantiles.values())
    maxima = [dict(question=int(q), maximum=float(scores[over & (question == q)].max()),
                   over_steps=int(np.count_nonzero(over & (question == q)))) for q in np.unique(question[over])]
    maxima.sort(key=lambda r: (-r['maximum'], r['question']))
    maximum_leave_question_out_relative_change = 1-maxima[1]['maximum']/maxima[0]['maximum']
    params = params_from_plan(read(ROOT/BASE/'configs/seal_comparison130_20260912/plan.json'))
    new_params = SimpleNamespace(**vars(params)); new_params.low_val_2 = -amplitude
    digest = hashlib.sha256(); boundary_rows = []
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tar:
        for q, line in enumerate(tar.extractfile('generations.jsonl')):
            digest.update(line); row = json.loads(line); p = np.exp(np.array(row['logprobs']))
            for pos, count, means, variance in step_statistics(row['token_ids'], np.column_stack([p, p]), set(params.boundary_token_ids), params.think_end_token_id):
                boundary_rows.append((q, pos, means[0], variance[0]))
    require(q == 499 and digest.hexdigest() == fit['calibration_source_sha256'], 'Calibration answers changed')
    data = np.array(boundary_rows); compute = load_controller()['compute_rebalance_coefficient']
    before = np.asarray(compute(tensor(data[:, 2], dtype=np.float32), tensor(data[:, 3], dtype=np.float32), params))
    after = np.asarray(compute(tensor(data[:, 2], dtype=np.float32), tensor(data[:, 3], dtype=np.float32), new_params))
    require(np.isfinite(after).all() and np.all(after >= before-2e-7), 'Reduced negative bound intensified negative control')
    require(np.array_equal(after[before >= 0], before[before >= 0]), 'Positive branch changed')
    affected = abs(after-before) >= .1; affected_questions = len(np.unique(data[affected, 0])); gate = plan['CPU_gate']
    passed = (ratio >= gate['minimum_reduction_in_aggressive_amplitude'] and
        stability >= gate['minimum_max_over_min_quantile_stability_inverse'] and
        affected.mean() >= gate['minimum_fraction_of_original_nonempty_boundary_coefficients_changed_by_at_least_point1'] and
        affected_questions >= gate['minimum_affected_questions'])
    out.mkdir(parents=True)
    arrays = out/'crossing_scores.npz'; np.savez_compressed(arrays, scores=scores, over=over, question=question,
        boundary_question=data[:, 0].astype(np.int32), boundary_position=data[:, 1].astype(np.int32), original_coefficient=before, robust_coefficient=after)
    result = dict(status='completed_CPU_robust_amplitude_calibration_not_generation',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True), source_sha256=source_hashes,
        original_vector_reproduced_bitwise=True, original_amplitude=original_amplitude, candidate_amplitude=amplitude,
        aggressive_amplitude_reduction=ratio, quantile=.95, quantile_method='higher', over_states=int(over.sum()),
        empirical_crossed_over_fraction=float(np.mean(scores[over] <= amplitude)), group_quantiles=group_quantiles,
        group_stability_ratio=stability, maximum_leave_one_question_out_relative_change=maximum_leave_question_out_relative_change,
        question_maxima=maxima, nonempty_boundaries=len(data), affected_boundaries=int(affected.sum()),
        affected_fraction=float(affected.mean()), affected_questions=affected_questions, original_positive_branch_unchanged=True,
        passes_fixed_gate=bool(passed), decision='prepare_fresh_GPU_screen' if passed else 'stop_no_GPU_no_quantile_search',
        arrays_sha256=sha(arrays), CPU_checks=nchecks, cpu_seconds=time.perf_counter()-started, GPU_calls=0, new_answers=0,
        limitations=['Empirical separator coverage is not correctness or compression.',
            'Original400/100 groups and fitted separator are development data; no conformal or independent-validation guarantee.'])
    if passed:
        candidate = copy.deepcopy(fit); candidate['version'] = 'auto-code-v2-robust-aggressive-q95'
        candidate['parameters']['low_val_2'] = -amplitude
        candidate['method'] = 'Originalraw direction and controller; only aggressive amplitude uses fixed empirical0.95 crossing quantile, floored at0.5.'
        candidate['robust_calibration'] = dict(hypothesis_sha256=result['plan_sha256'], original_fit_sha256=source_hashes['fit.json'],
            score_arrays_sha256=result['arrays_sha256'], quantile=.95, method='higher', all500_development=True)
        shutil.copyfile(backup/'auto_vector.pt', out/'auto_vector.pt'); save(out/'fit.json', candidate)
        result['candidate_files_sha256'] = {name: sha(out/name) for name in ['fit.json', 'auto_vector.pt']}
    save(out/'audit.json', result)
    print({k: result[k] for k in ['status', 'original_amplitude', 'candidate_amplitude', 'aggressive_amplitude_reduction', 'group_stability_ratio', 'maximum_leave_one_question_out_relative_change', 'affected_fraction', 'affected_questions', 'passes_fixed_gate', 'decision', 'cpu_seconds']})


if __name__ == '__main__': main()
