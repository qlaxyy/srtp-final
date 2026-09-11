"""Count positive-branch opportunities on saved traces; no benefit estimate."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from audit_control_alignment import load_controller, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Immutable CPU result exists')
    plan_path = ROOT/BASE/'configs/positive_branch_ablation_20260912.json'
    plan = read(plan_path)
    source = read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']
    backup = ROOT/source['inputs']['backup']
    for name, digest in source['cpu_result']['inputs_sha256'].items():
        require(sha(backup/name) == digest, 'Original asset changed: '+name)
    steps = read(backup/'steps.json')
    support_path = ROOT/'.codex_work/overnight_research_20260912/control_point_prepared/matched_mask.npy'
    support = np.load(support_path)
    q = np.array([s['question'] for s in steps])
    indices = np.flatnonzero(support)-1
    require(np.array_equal(q[indices], q[indices+1]), 'Boundary crossed question')
    confidence = np.array([s['confidence'] for s in steps])
    variance = np.array([s['variance'] for s in steps])
    length = np.array([s['stop']-s['start'] for s in steps])
    params = SimpleNamespace(**read(backup/'fit.json')['parameters'])
    compute = load_controller()['compute_rebalance_coefficient']
    coefficient = np.asarray(compute(tensor(confidence), tensor(variance), params))[indices]
    positive = coefficient > 0
    negative = coefficient < 0
    candidate = np.minimum(coefficient, 0)
    require(np.isfinite(coefficient).all(), 'Nonfinite coefficient')
    require(np.array_equal(candidate[negative], coefficient[negative]), 'Negative branch changed')
    require(not np.any(candidate > 0), 'Positive branch survived')
    gate = plan['cpu_gate']
    fraction = float(positive.mean())
    question_count = len(np.unique(q[indices][positive]))
    summaries = {}
    for name, mask in [('positive', positive), ('negative', negative)]:
        values = indices[mask]
        summaries[name] = dict(count=len(values), questions=len(np.unique(q[values])),
            mean_confidence=float(confidence[values].mean()),
            mean_variance=float(variance[values].mean()),
            mean_preceding_step_tokens=float(length[values].mean()),
            median_preceding_step_tokens=float(np.median(length[values])),
            coefficient_quantiles=np.quantile(coefficient[mask], [0, .25, .5, .75, 1]).tolist())
    passed = (fraction >= gate['minimum_positive_fraction_of_valid_saved_boundaries']
              and question_count >= gate['minimum_questions_with_positive_boundary'])
    result = dict(status='cpu_branch_opportunity_check_only',
        mechanism_plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        saved_boundaries=len(indices), positive_fraction=fraction,
        summaries=summaries, negative_coefficients_bitwise_preserved=True,
        passes_opportunity_gate=passed,
        limitation='Unsteered calibration trajectories only. These counts do not predict how often the online dynamic candidate will intervene or whether it will shorten generation.')
    save(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
