"""Describe confidence versus surface form without treating either as safety."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from replay_cycle_monitor import token_bytes


def classify(piece):
    text = piece.decode('utf-8', errors='replace').strip()
    if re.fullmatch('[0-9]+', text): return 0
    if re.search('[A-Za-z]', text): return 1
    if re.search('[0-9]', text): return 2
    if not text: return 3
    if any(ord(c) > 127 for c in text): return 4
    return 5


def fit_and_evaluate(features, target, train, development, question):
    def weights(mask):
        _, inverse, counts = np.unique(question[mask], return_inverse=True, return_counts=True)
        w = 1./counts[inverse]
        return w * mask.sum()/w.sum()
    wt, wd = weights(train), weights(development)
    xt, xd, yt, yd = features[train], features[development], target[train], target[development]
    mean = np.average(xt, axis=0, weights=wt)
    scale = np.sqrt(np.average((xt-mean)**2, axis=0, weights=wt))
    scale = np.where(scale > 1e-12, scale, 1.)
    zt, zd = (xt-mean)/scale, (xd-mean)/scale
    mt = float(np.average(yt, weights=wt))
    slopes = np.linalg.solve(zt.T@(wt[:, None]*zt)+np.eye(zt.shape[1]), zt.T@(wt*(yt-mt)))
    pred = zd@slopes+mt
    mse = float(np.average((yd-pred)**2, weights=wd))
    null = float(np.average((yd-mt)**2, weights=wd))
    return dict(training_intercept=mt, standardized_slopes=slopes.tolist(),
                training_feature_mean=mean.tolist(), training_feature_scale=scale.tolist(),
                development_weighted_MSE=mse, training_mean_null_MSE=null,
                development_R2_against_training_mean=1-mse/null,
                development_prediction_range=[float(pred.min()), float(pred.max())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); require(not a.output.exists(), 'Immutable output exists')
    started = time.perf_counter()
    plan_path = ROOT/BASE/'configs/confidence_surface_audit_20260912.json'
    plan = read(plan_path)
    replay = read(ROOT/'.codex_work/overnight_research_20260912/control_point_prepared/replay_plan.json')
    backup = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    tokenizer_path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    require(sha(tokenizer_path) == replay['model_files_sha256']['tokenizer.json'], 'Tokenizer changed')
    require(sha(backup/'steps.json') == replay['original_files_sha256']['steps.json'], 'Steps changed')
    pieces = token_bytes(read(tokenizer_path))
    categories = np.full(max(pieces)+1, -1, dtype=np.int8)
    for token, piece in pieces.items(): categories[token] = classify(piece)
    require([classify(s) for s in [b' 12', b'\\frac', b'.25', b'\n', 'α'.encode(), b'+']] == list(range(6)), 'Category order failed')
    steps = read(backup/'steps.json'); selection = read(backup/'selected_layer.json')
    require(len(steps) == 84008, 'Wrong step count')
    grouped = [[] for _ in range(500)]
    for i, step in enumerate(steps): grouped[step['question']].append((i, step))
    features = np.empty((len(steps), 7)); confidence = np.array([s['confidence'] for s in steps])
    questions = np.array([s['question'] for s in steps]); counts = np.zeros((500, 6), dtype=np.int64)
    sums = np.zeros((500, 6)); max_error = 0.; capped = 0; digest = hashlib.sha256()
    archive = ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive) as tar:
        for q, line in enumerate(tar.extractfile('generations.jsonl')):
            digest.update(line); row = json.loads(line)
            ids = np.array(row['token_ids']); probabilities = np.exp(np.array(row['logprobs']))
            require(len(ids) == len(probabilities) and np.all((probabilities > 0)&(probabilities <= 1)), 'Invalid saved probabilities')
            capped += row['finish_reason'] == 'length'
            for i, step in grouped[q]:
                begin, end = step['start'], step['stop']; require(0 <= begin < end <= len(ids), 'Wrong interval')
                cls = categories[ids[begin:end]]; require(np.all(cls >= 0), 'Unknown token')
                prob = probabilities[begin:end]; n = end-begin
                max_error = max(max_error, abs(float(prob.mean())-confidence[i]))
                count = np.bincount(cls, minlength=6); sums[q] += np.bincount(cls, weights=prob, minlength=6)
                counts[q] += count; features[i] = np.r_[np.log1p(n), count/n]
    require(q == 499 and digest.hexdigest() == replay['generations_sha256'], 'Saved answers changed')
    require(max_error < 1e-10, 'Original step confidence does not align')
    masks = {k: np.isin(questions, selection[k]) for k in ['training_questions', 'validation_questions']}
    require(np.all(masks['training_questions'] ^ masks['validation_questions']), 'Question groups overlap')
    regressions = {name: fit_and_evaluate(features[:, columns], confidence,
                    masks['training_questions'], masks['validation_questions'], questions)
                   for name, columns in [('length_only', [0]), ('length_and_surface', list(range(7)))]}
    statistics = {}
    for c, name in enumerate(plan['token_classes_order']):
        present = counts[:, c] > 0
        statistics[name] = dict(count=int(counts[:, c].sum()), questions=int(present.sum()),
            pooled_mean_probability=float(sums[:, c].sum()/counts[:, c].sum()) if present.any() else None,
            question_equal_mean_probability=float(np.mean(sums[present, c]/counts[present, c])) if present.any() else None,
            question_equal_fraction=float(np.mean(counts[:, c]/counts.sum(axis=1))))
    result = dict(status='completed_CPU_descriptive_audit_no_controller_change',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        generations_sha256=digest.hexdigest(), steps_sha256=sha(backup/'steps.json'),
        tokenizer_sha256=sha(tokenizer_path), split_sha256=sha(backup/'selected_layer.json'),
        questions=500, capped_answers_included=capped, steps=len(steps), step_tokens=int(counts.sum()),
        maximum_original_mean_reconstruction_error=max_error, token_class_statistics=statistics,
        regressions=regressions, development_MSE_improvement_over_length_only=1-
            regressions['length_and_surface']['development_weighted_MSE']/regressions['length_only']['development_weighted_MSE'],
        interpretation=plan['interpretation'], cpu_seconds=time.perf_counter()-started,
        model_loads=0, new_answers=0, GPU_calls=0)
    save(a.output, result)
    print(json.dumps({k: result[k] for k in ['status', 'questions', 'capped_answers_included', 'step_tokens', 'maximum_original_mean_reconstruction_error', 'regressions', 'development_MSE_improvement_over_length_only', 'cpu_seconds']}, indent=2))


if __name__ == '__main__': main()
