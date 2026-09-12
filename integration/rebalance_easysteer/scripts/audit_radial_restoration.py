"""One fixed CPU opportunity check on saved states; never loads a model."""
import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_control_alignment import load_controller, tensor
from mechanism_candidates import (ROOT, BASE, bfloat16, read, read_vector,
                                  require, save, sha)


def restore(x, displacement):
    """Retain the original additive direction and current complete-state norm.

    Undefined direction at zero norm uses the original state as a no-op.
    This NumPy reference does not simulate the compiled inference graph.
    """
    y = x + displacement
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    valid = (nx > 1e-12) & (ny > 1e-12)
    scale = np.divide(nx, ny, out=np.ones_like(nx), where=valid)
    return np.where(valid, y * scale, x)


def checks():
    x = np.array([[3., 4., 0.], [0., 0., 0.], [3., 4., 0.],
                  [3., 4., 0.], [3., 4., 0.]])
    d = np.array([[0., 0., 2.], [1., 0., 0.], [-3., -4., 0.],
                  [0., 0., 0.], [-6., -8., 0.]])
    y = restore(x, d)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1),
                               np.linalg.norm(x, axis=1), atol=1e-14)
    np.testing.assert_array_equal(y[1:4], x[1:4])
    np.testing.assert_allclose(y[0], (x[0]+d[0])*5/np.sqrt(29))
    np.testing.assert_array_equal(y[4], -x[4])
    rounded = bfloat16(np.array([1., 1.+1/256, 1.+3/256, -1.-1/256]))
    np.testing.assert_array_equal(rounded, [1., 1., 1.+1/64, -1.])
    return 5


def distribution(v):
    return dict(mean=float(np.mean(v)),
                quantiles=np.quantile(v, [0, .1, .25, .5, .75, .9, .99, 1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Immutable output already exists')
    started = time.perf_counter()
    nchecks = checks()
    plan_path = ROOT / BASE / 'configs/radial_restoration_20260912.json'
    plan = read(plan_path)
    p = plan['inputs']
    backup = ROOT / p['backup']
    paths = {'hidden': ROOT/p['hidden_path'], 'steps': backup/'steps.json',
             'vector': backup/'auto_vector.pt', 'fit': backup/'fit.json'}
    for name, path in paths.items():
        require(sha(path) == p[name+'_sha256'], name+' changed')
    replay_plan = read(paths['hidden'].parent/'plan.json')
    mask_path = ROOT / p['support_path']
    require(sha(mask_path) == replay_plan['matched_mask_sha256'], 'Support changed')
    steps = read(paths['steps'])
    support = np.load(mask_path)
    q_all = np.array([s['question'] for s in steps])
    current = np.flatnonzero(support)
    previous = current-1
    require(len(current) == 83508 and len(steps) == 84008, 'Unexpected support')
    require(np.array_equal(q_all[current], q_all[previous]), 'Cross-question shift')
    require(all(steps[i]['start'] > 0 and
                steps[i-1]['stop'] <= steps[i]['start']-1 for i in current),
            'Preceding evidence overlaps current boundary')
    gaps = [int(i) for i in current if steps[i-1]['stop'] < steps[i]['start']-1]
    gap_q = {steps[i]['question'] for i in gaps}
    frozen = read(ROOT/BASE/'configs/final_results_20260909.json')
    boundaries = set(frozen['benchmarks'][0]['protocol']['dynamic_params']['boundary_token_ids'])
    # The online ready mask requires counts>0, so empty steps retain the most
    # recent nonempty step's coefficient. Check the actual skipped token IDs.
    gap_tokens = {}
    digest = hashlib.sha256()
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as archive:
        for question, line in enumerate(archive.extractfile('generations.jsonl')):
            digest.update(line)
            if question in gap_q:
                gap_tokens[question] = json.loads(line)['token_ids']
    require(digest.hexdigest() == replay_plan['generations_sha256'], 'Saved answers changed')
    for i in gaps:
        ids = gap_tokens[steps[i]['question']]
        require(all(t in boundaries for t in ids[steps[i-1]['stop']:steps[i]['start']]),
                'Gap includes unaccounted content')
    groups = read(backup/'selected_layer.json')
    group_masks = {key: np.isin(q_all[current], groups[key])
                   for key in ['training_questions', 'validation_questions']}
    require(np.all(group_masks['training_questions'] ^ group_masks['validation_questions']),
            'Question split missing or overlapping')
    params = SimpleNamespace(**read(paths['fit'])['parameters'])
    c = load_controller()['compute_rebalance_coefficient'](
        tensor([s['confidence'] for s in steps]),
        tensor([s['variance'] for s in steps]), params)[previous]
    c = np.asarray(c, dtype=np.float64)
    require(np.isfinite(c).all(), 'Nonfinite controller')
    d = read_vector(paths['vector'], paths['vector']).astype(np.float64)
    states = np.load(paths['hidden'], mmap_mode='r')
    require(states.shape == (84008, 1536), 'Wrong state shape')
    fields = {name: np.empty(len(current), dtype=np.float64) for name in
              ['state_norm', 'norm_ratio', 'correction_vs_displacement',
               'cosine_with_direction', 'angle_radians',
               'restored_bf16_relative_norm_error', 'restored_bf16_rms_change']}
    max_direction_error = max_norm_error = 0.
    for start in range(0, len(current), 1024):
        sl = slice(start, start+1024)
        x = np.asarray(states[current[sl]], dtype=np.float64)
        delta = c[sl, None] * d[None, :]
        additive = x + delta
        restored = restore(x, delta)
        nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(additive, axis=1)
        require(np.isfinite(restored).all() and np.all(nx > 0) and np.all(ny > 0),
                'Degenerate saved geometry')
        max_norm_error = max(max_norm_error, float(np.max(abs(
            np.linalg.norm(restored, axis=1)/nx-1))))
        max_direction_error = max(max_direction_error, float(np.max(abs(
            restored/nx[:, None]-additive/ny[:, None]))))
        norm_delta = abs(c[sl])*np.linalg.norm(d)
        fields['state_norm'][sl] = nx
        fields['norm_ratio'][sl] = ny/nx
        fields['correction_vs_displacement'][sl] = np.divide(abs(ny-nx), norm_delta,
            out=np.zeros_like(nx), where=norm_delta > 0)
        fields['cosine_with_direction'][sl] = (x@d)/(nx*np.linalg.norm(d))
        fields['angle_radians'][sl] = np.arccos(np.clip(
            np.sum(x*additive, axis=1)/(nx*ny), -1, 1))
        rounded = bfloat16(restored)
        fields['restored_bf16_relative_norm_error'][sl] = abs(
            np.linalg.norm(rounded.astype(np.float64), axis=1)/nx-1)
        fields['restored_bf16_rms_change'][sl] = np.sqrt(np.mean(
            (rounded.astype(np.float64)-bfloat16(additive).astype(np.float64))**2, axis=1))
    gate = plan['cpu_gate']
    affected = abs(fields['norm_ratio']-1) >= gate['minimum_absolute_norm_ratio_change']
    q = q_all[current]
    masks = {'all': np.ones(len(current), dtype=bool), 'positive': c > 0,
             'negative': c < 0, **group_masks}
    summary = {}
    for name, mask in masks.items():
        require(np.any(mask), 'Empty analysis group '+name)
        summary[name] = dict(count=int(mask.sum()), questions=len(np.unique(q[mask])),
            affected_count=int((affected & mask).sum()),
            affected_fraction=float(affected[mask].mean()),
            affected_questions=len(np.unique(q[affected & mask])),
            fields={k: distribution(v[mask]) for k, v in fields.items()})
    passed = (summary['all']['affected_fraction'] >= gate['minimum_affected_fraction']
        and summary['all']['affected_questions'] >= gate['minimum_affected_questions']
        and all(summary[key]['affected_fraction'] >=
            gate['minimum_affected_fraction_in_each_original_question_group'] for key in group_masks)
        and max_norm_error < 1e-12 and max_direction_error < 1e-12)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'boundaries.npz', current_row=current, previous_row=previous,
                        question=q, coefficient=c, affected=affected, **fields)
    result = dict(status='completed_cpu_geometry_only',
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        input_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths.values()},
        support_sha256=sha(mask_path), split_sha256=sha(backup/'selected_layer.json'),
        consecutive_boundary_rows=gaps, saved_answers_sha256=digest.hexdigest(),
        checks=nchecks, max_direction_error=max_direction_error,
        max_relative_restored_norm_error=max_norm_error, summary=summary,
        passes_fixed_cpu_gate=bool(passed),
        decision='prepare_separate_runtime_plan' if passed else 'stop_no_GPU_no_retuning',
        boundaries_sha256=sha(a.output/'boundaries.npz'),
        cpu_seconds=time.perf_counter()-started, model_loads=0, new_answers=0, GPU_calls=0,
        limitations=['Unsteered saved trajectories and offline FP64 controller; not online incidence.',
                     'BF16 round-to-even emulation is not a compiled-kernel or model-forward check.',
                     'Norm restoration changes later residual mixing even when immediate RMSNorm is nearly invariant.',
                     'Question groups were used in original layer selection; development only.'])
    save(a.output/'summary.json', result)
    print({k: result[k] for k in ['status','passes_fixed_cpu_gate','decision','cpu_seconds']})
    print({k: {f: v[f] for f in ['count','affected_fraction','affected_questions']}
           for k,v in summary.items()})


if __name__ == '__main__':
    main()
