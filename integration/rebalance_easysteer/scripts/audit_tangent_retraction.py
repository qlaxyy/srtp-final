"""Check one fixed angular opportunity on saved unsteered geometry."""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from mechanism_candidates import ROOT, BASE, read, read_vector, require, save, sha


def angles(t, rho):
    transverse = abs(t) * np.sqrt(np.maximum(0., 1. - rho*rho))
    return np.arctan2(transverse, 1. + t*rho), np.arctan(transverse)


def explicit(x, z):
    nx2 = np.sum(x*x, axis=-1, keepdims=True)
    radial = np.divide(np.sum(x*z, axis=-1, keepdims=True), nx2,
                       out=np.zeros_like(nx2), where=nx2 > 0)
    tangent = z - x*radial
    y = x + tangent
    scale = np.divide(np.sqrt(nx2), np.linalg.norm(y, axis=-1, keepdims=True),
                      out=np.ones_like(nx2), where=nx2 > 0)
    return np.where(nx2 > 0, y*scale, x), tangent


def checks():
    # Direct 2-D vector arithmetic independently checks both scalar formulas.
    t, rho = np.meshgrid(np.array([-3., -1.5, -.5, 0., .1, .5, 1.5, 3.]),
                         np.array([-1., -.8, -.1, 0., .1, .8, 1.]))
    t, rho = t.ravel(), rho.ravel()
    x = np.column_stack([np.ones(len(t)), np.zeros(len(t))])
    z = t[:, None] * np.column_stack([rho, np.sqrt(1-rho*rho)])
    restored, tangent = explicit(x, z)
    before, after = angles(t, rho)
    np.testing.assert_allclose(before, np.arctan2(abs(z[:, 1]), 1+z[:, 0]), atol=1e-14)
    np.testing.assert_allclose(after, np.arctan2(abs(restored[:, 1]), restored[:, 0]), atol=1e-14)
    np.testing.assert_allclose(np.sum(tangent*x, axis=1), 0., atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(restored, axis=1), 1., atol=1e-14)
    np.testing.assert_array_equal(restored[abs(rho) == 1], x[abs(rho) == 1])
    np.testing.assert_array_equal(restored[t == 0], x[t == 0])
    zero, _ = explicit(np.zeros((1, 2)), np.array([[1., 2.]]))
    np.testing.assert_array_equal(zero, np.zeros((1, 2)))
    return dict(explicit_cases=len(t)+1, assertions=7)


def distribution(values):
    return dict(mean=float(np.mean(values)),
                quantiles=np.quantile(values, [0, .1, .25, .5, .75, .9, .99, 1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Immutable output exists')
    started = time.perf_counter()
    check_result = checks()
    plan_path = ROOT/BASE/'configs/tangent_retraction_20260912.json'
    plan = read(plan_path)
    base = ROOT/'.codex_work/overnight_research_20260912/radial_restoration_cpu'
    original = read(base/'summary.json')
    array_path = base/'boundaries.npz'
    require(sha(array_path) == original['boundaries_sha256'], 'Geometry changed')
    backup = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    require(sha(backup/'selected_layer.json') == original['split_sha256'], 'Groups changed')
    selection = read(backup/'selected_layer.json')
    data = np.load(array_path)
    c, norm, rho, q = [data[k] for k in ['coefficient', 'state_norm', 'cosine_with_direction', 'question']]
    require(len(q) == 83508 and np.all(norm > 0) and np.all(abs(rho) <= 1), 'Invalid support')
    vector = read_vector(backup/'auto_vector.pt', backup/'auto_vector.pt').astype(np.float64)
    t = c*np.linalg.norm(vector)/norm
    before, after = angles(t, rho)
    saved_error = float(np.max(abs(before-data['angle_radians'])))
    require(saved_error < 1e-7, 'Independent original angle mismatch')
    # Verify the saved population against explicit 2-D tangent projection too.
    x = np.column_stack([norm, np.zeros(len(norm))])
    z = norm[:, None]*t[:, None]*np.column_stack([rho, np.sqrt(1-rho*rho)])
    restored, tangent = explicit(x, z)
    formula_error = float(np.max(abs(after-np.arctan2(abs(restored[:, 1]), restored[:, 0]))))
    norm_error = float(np.max(abs(np.linalg.norm(restored, axis=1)/norm-1)))
    orthogonality_error = float(np.max(abs(np.sum(x*tangent, axis=1))/norm**2))
    finite = bool(np.isfinite(restored).all() and np.isfinite(before).all() and np.isfinite(after).all())
    require(finite and max(formula_error, norm_error, orthogonality_error) < 1e-12, 'Geometry identity failed')
    groups = {k: np.isin(q, selection[k]) for k in ['training_questions', 'validation_questions']}
    require(np.all(groups['training_questions'] ^ groups['validation_questions']), 'Groups overlap')
    change = after-before
    gate = plan['CPU_gate']
    affected = abs(change) >= gate['minimum_absolute_angle_difference_radians']
    summary = {}
    fields = dict(original_angle=before, tangent_angle=after, angle_change=change,
                  relative_tangent_displacement=np.linalg.norm(tangent, axis=1)/norm,
                  retained_displacement_fraction=np.sqrt(1-rho*rho))
    for name, mask in {'all': np.ones(len(q), dtype=bool), 'positive': c > 0, 'negative': c < 0, **groups}.items():
        summary[name] = dict(count=int(mask.sum()), questions=len(np.unique(q[mask])),
            affected_count=int((mask & affected).sum()), affected_fraction=float(affected[mask].mean()),
            affected_questions=len(np.unique(q[mask & affected])),
            angle_increased_count=int((mask & (change > 1e-12)).sum()),
            angle_decreased_count=int((mask & (change < -1e-12)).sum()),
            fields={k: distribution(v[mask]) for k, v in fields.items()})
    passed = (summary['all']['affected_fraction'] >= gate['minimum_affected_fraction_all_boundaries']
        and summary['all']['affected_questions'] >= gate['minimum_affected_questions']
        and all(summary[k]['affected_fraction'] >= gate['minimum_affected_fraction_each_group'] for k in groups))
    result = dict(status='completed_CPU_geometry_only',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        geometry_sha256=sha(array_path), split_sha256=sha(backup/'selected_layer.json'),
        vector_sha256=sha(backup/'auto_vector.pt'), checks=check_result,
        saved_original_angle_error=saved_error, tangent_formula_error=formula_error,
        restored_norm_error=norm_error, tangent_orthogonality_error=orthogonality_error,
        summary=summary, passes_fixed_cpu_gate=bool(passed),
        decision='prepare_separate_runtime_plan' if passed else 'stop_no_GPU_no_retuning',
        cpu_seconds=time.perf_counter()-started, model_loads=0, new_answers=0, GPU_calls=0,
        limitations=plan['limitations'])
    save(a.output, result)
    print({k: result[k] for k in ['status', 'passes_fixed_cpu_gate', 'decision', 'cpu_seconds']})
    print({k: {f: v[f] for f in ['affected_count', 'affected_fraction', 'affected_questions']} for k, v in summary.items()})


if __name__ == '__main__':
    main()
