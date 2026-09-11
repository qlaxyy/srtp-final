"""Grouped CPU screening of actual prefix-answer correctness readouts."""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require


def fit_ridge(features, labels, questions):
    unique, counts = np.unique(questions, return_counts=True)
    inverse = dict(zip(unique.tolist(), (1/counts).tolist()))
    weights = np.array([inverse[q] for q in questions])
    weights /= weights.mean()
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(np.average((features-mean)**2, axis=0,
                                         weights=weights), 1e-12))
    z = (features-mean)/scale/np.sqrt(features.shape[1])
    intercept = float(np.average(labels, weights=weights))
    root_weights = np.sqrt(weights)
    design = z*root_weights[:, None]
    target = (labels-intercept)*root_weights
    dual = np.linalg.solve(design@design.T+np.eye(len(design)), target)
    coefficient = design.T@dual
    return dict(mean=mean, scale=scale, coefficient=coefficient,
                intercept=intercept, alpha=1., weights=weights)


def predict(model, features):
    z = (features-model['mean'])/model['scale']/np.sqrt(features.shape[1])
    return z@model['coefficient']+model['intercept']


def auc(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None]-negative[None, :]
    return float(((difference > 0)+.5*(difference == 0)).mean())


def test_ridge_against_primal_and_fixed_training_transform():
    x = np.array([[0., 1.], [1., 0.], [2., 3.], [4., 2.]])
    y = np.array([0., 0., 1., 1.])
    q = np.array([0, 0, 1, 2])
    model = fit_ridge(x, y, q)
    z = (x-model['mean'])/model['scale']/np.sqrt(2)
    w = model['weights']
    primal = np.linalg.solve(z.T@(w[:, None]*z)+np.eye(2),
                             z.T@(w*(y-model['intercept'])))
    np.testing.assert_allclose(model['coefficient'], primal, atol=1e-12)
    heldout = np.array([[1., 2.], [3., 4.]])
    expected = predict(model, heldout)
    with_outlier = predict(model, np.vstack([heldout, [1e6, -1e6]]))
    np.testing.assert_array_equal(expected, with_outlier[:2])
    assert auc(np.array([0, 1]), np.array([.5, .5])) == .5
    assert auc(np.array([0, 1]), np.array([0., 1.])) == 1.
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), 'Immutable CPU readout result already exists')
    plan_path = ROOT/BASE/'configs/causal_correctness_readout_20260912.json'
    plan = read(plan_path)
    inputs = {name:ROOT/value for name,value in plan['inputs'].items()}
    prefix_root = inputs['trial_analysis'].parents[1]
    manifest = read(prefix_root/'prefix_probe_all_20260912.manifest.json')
    for name in ['trial_analysis', 'trial_records']:
        require(sha(inputs[name]) == manifest[inputs[name].relative_to(prefix_root).as_posix()], 'Trial evidence changed')
    replay = read(inputs['hidden'].parent/'ledger.json')
    require(sha(inputs['hidden']) == replay['files_sha256'][inputs['hidden'].name], 'Hidden states changed')
    root_record = read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']
    require(sha(inputs['steps']) == root_record['cpu_result']['inputs_sha256']['steps.json'], 'Step alignment changed')
    analysis = read(inputs['trial_analysis'])
    probes = read(inputs['trial_records'])['records']
    require(len(analysis['per_question']) == 32 and len(probes) == 327, 'Wrong causal dataset')
    steps = read(inputs['steps'])
    lookup = {(row['question'], row['start']):i for i,row in enumerate(steps)}
    require(len(lookup) == len(steps), 'Duplicate state position')
    labels = {(q['calibration_index'], k):int(correct)
              for q in analysis['per_question']
              for k,correct in zip(q['checkpoints'], q['trial_correct'], strict=True)}
    joined, missing = [], []
    for probe in probes:
        key = (probe['calibration_index'], probe['prefix_tokens'])
        if key not in lookup:
            missing.append(dict(question=key[0], prefix=key[1]))
            continue
        index = lookup[key]
        require(index > 0 and steps[index-1]['question'] == key[0], 'Noncausal incoming statistics')
        joined.append(dict(question=key[0], prefix=key[1], hidden_row=index,
                           label=labels[key], confidence=steps[index-1]['confidence'],
                           variance=steps[index-1]['variance']))
    question_ids = [r['calibration_index'] for r in analysis['per_question']]
    train_ids = random.Random(plan['split']['seed']).sample(question_ids, 24)
    heldout_ids = [q for q in question_ids if q not in train_ids]
    q = np.array([r['question'] for r in joined])
    y = np.array([r['label'] for r in joined], dtype=float)
    train = np.isin(q, train_ids)
    heldout = ~train
    require(not set(train_ids)&set(heldout_ids) and len(heldout_ids) == 8, 'Question leakage')
    hidden = np.load(inputs['hidden'], mmap_mode='r')
    require(hidden.shape == (84008, 1536), 'Wrong hidden feature matrix')
    features = dict(hidden=np.asarray(hidden[[r['hidden_row'] for r in joined]], dtype=float),
                    length=np.array([[r['prefix']] for r in joined], dtype=float),
                    confidence_variance=np.array([[r['confidence'], r['variance']] for r in joined]))
    gate = plan['cpu_gate']
    support = {split:{str(label):len(np.unique(q[mask & (y == label)])) for label in [0, 1]}
               for split,mask in [('training', train), ('heldout', heldout)]}
    basic = dict(join_coverage=len(joined)/327 >= gate['minimum_join_coverage'],
                 training_support=min(support['training'].values()) >= gate['minimum_training_questions_per_label'],
                 heldout_support=min(support['heldout'].values()) >= gate['minimum_holdout_questions_per_label'])
    out.mkdir(parents=True)
    models = {}
    for name,x in features.items():
        require(np.isfinite(x).all(), 'Nonfinite features')
        model = fit_ridge(x[train], y[train], q[train])
        train_scores = predict(model, x[train])
        scores = predict(model, x[heldout])
        threshold = float(np.nextafter(train_scores[y[train] == 0].max(), np.inf))
        accepted = scores >= threshold
        item = dict(heldout_auc=auc(y[heldout], scores), threshold=threshold,
            training_auc=auc(y[train], train_scores),
            training_false_acceptances=int(((train_scores >= threshold)&(y[train] == 0)).sum()),
            heldout_false_acceptances=int((accepted & (y[heldout] == 0)).sum()),
            heldout_accepted=int(accepted.sum()), heldout_acceptance_fraction=float(accepted.mean()),
            heldout_questions_with_acceptance=len(np.unique(q[heldout][accepted])),
            heldout_scores=scores.tolist(), heldout_accepted_flags=accepted.tolist())
        np.savez(out/(name+'.npz'), **model)
        item['asset_sha256'] = sha(out/(name+'.npz'))
        models[name] = item
    candidate = models['hidden']
    best_control = max(models[n]['heldout_auc'] for n in ['length', 'confidence_variance'])
    checks = dict(basic,
        auc=candidate['heldout_auc'] >= gate['minimum_holdout_auc'],
        control_advantage=candidate['heldout_auc']-best_control >= gate['minimum_auc_gain_over_best_fixed_control'],
        false_acceptances=candidate['heldout_false_acceptances'] <= gate['maximum_holdout_false_acceptances'],
        coverage=candidate['heldout_acceptance_fraction'] >= gate['minimum_holdout_acceptance_fraction'],
        question_coverage=candidate['heldout_questions_with_acceptance'] >= gate['minimum_holdout_questions_with_acceptance'])
    result = dict(status='completed_cpu_development_screen', plan_sha256=sha(plan_path, source=True),
        input_sha256={name:sha(path) for name,path in inputs.items()}, script_sha256=sha(Path(__file__), source=True),
        training_questions=train_ids, heldout_questions=heldout_ids, matched_rows=len(joined), missing_rows=missing,
        label_question_support=support, training_rows=int(train.sum()), heldout_rows=int(heldout.sum()),
        heldout_question_ids=q[heldout].tolist(), heldout_labels=y[heldout].astype(int).tolist(),
        models=models, checks=checks, passes_fixed_cpu_gate=all(checks.values()),
        cpu_math_checks=test_ridge_against_primal_and_fixed_training_transform(),
        decision='Prepare stronger independent validation only' if all(checks.values()) else 'Stop this readout; no layer/regularization/threshold tuning',
        limitation=plan['limitations'], model_calls=0, new_answers=0)
    save(out/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['models', 'heldout_question_ids', 'heldout_labels', 'limitation']}))
    print(json.dumps({name:{k:v for k,v in values.items() if k not in ['heldout_scores','heldout_accepted_flags']} for name,values in models.items()}))


if __name__ == '__main__':
    main()
