"""A feasible toy sampling path exposes confidence/variance coupling."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mechanism_candidates import ROOT, read, save, sha, require
from audit_control_alignment import load_controller, tensor, scalar_reference


def audit():
    fit = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/fit.json'
    params = SimpleNamespace(**read(fit)['parameters'])
    # Two single-content-token steps. Both select token1. This path has
    # nonzero support even after the fixed temperature0.7/top_p0.95 sampler.
    raw = np.array([[.88, .119999, .000001], [.78, .12, .10]], dtype=np.float64)
    maximum = raw.max(axis=1); selected = raw[:, 1]
    allowed = []; conditional_selected_probability = []
    for probabilities in raw:
        scaled = probabilities**(1/.7); scaled /= scaled.sum()
        order = np.argsort(-scaled)
        count = int(np.searchsorted(np.cumsum(scaled[order]), .95))+1
        retained = order[:count]
        require(1 in retained, 'Hypothetical selected token is outside the sampling support')
        allowed.append(retained.tolist())
        conditional_selected_probability.append(float(scaled[1]/scaled[retained].sum()))
    modal_v = float((maximum[1]-maximum[0])**2/4)
    selected_v = float((selected[1]-selected[0])**2/4)
    namespace = load_controller(); compute = namespace['compute_rebalance_coefficient']
    settings = [('original_modal', float(maximum[1]), modal_v),
                ('selected_mean_with_old_variance_counterfactual', float(selected[1]), modal_v),
                ('selected_mean_and_derived_variance_candidate', float(selected[1]), selected_v)]
    constants = namespace['_curve_constants'](params.q25c, params.q75c, params.low_val_1,
                                            'cpu', np.dtype('float64'), params.curve_tau)
    rows = []
    for label, confidence, variance in settings:
        actual = float(compute(tensor([confidence], dtype=np.float32),
                               tensor([variance], dtype=np.float32), params)[0])
        reference = scalar_reference(confidence, variance, params, constants)
        require(abs(actual-reference['coefficient']) < .001, 'Independent scalar mismatch')
        rows.append(dict(label=label, confidence=confidence, variance=variance,
                         coefficient=actual, scalar_decomposition=reference))
    require(selected[1] < maximum[1] and rows[2]['coefficient'] > rows[0]['coefficient']+.5,
            'Expected counterexample disappeared')
    return dict(status='CPU_feasible_sampling_counterexample_not_model_trajectory',
        raw_distributions=raw.tolist(), chosen_token=1, temperature=.7, top_p=.95,
        retained_token_ids=allowed, sampling_law_probability_of_chosen_token=conditional_selected_probability,
        controller_cases=rows,
        conclusion='Lower realized-token probability does not universally produce a more negative coefficient: changing the step means also changes the two-step variance and can remove the strong negative adjustment.',
        impact='The fixed replay gate measures absolute control differences in either direction. Its pass is an opportunity check, not evidence for more compression. Keep all settings and gates unchanged.',
        limits='Constructed three-token distributions and one-content-token steps, not observed model frequencies, semantic correctness, or generated compression benefit. Counterfactual old variance is explanation only and is not another candidate.',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p, source=True) for p in
            [fit, Path(__file__), ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py']},
        model_loads=0, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable audit exists')
    result = audit(); save(args.output, result)
    print(dict(status=result['status'], coefficients={row['label']:row['coefficient']
        for row in result['controller_cases']}, GPU_calls=0))


if __name__ == '__main__':
    main()
