"""Synthetic contract checks only: no models, experiment data, or GPU imports."""
import json
import math
from pathlib import Path


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def paper_eat(values, alpha=0.2):
    mean = variance = 0.0
    result = []
    for n, value in enumerate(values, 1):
        mean = (1-alpha)*mean + alpha*value
        variance = (1-alpha)*variance + alpha*(value-mean)**2
        result.append(variance/(1-(1-alpha)**n))
    return result


def public_eat(values, alpha=0.2):
    # Audited public code initializes mean at the first observation and omits
    # Algorithm 1's variance debiasing; normalize is unused in that function.
    mean, variance = values[0], 0.0
    result = [variance]
    for value in values[1:]:
        mean = (1-alpha)*mean + alpha*value
        variance = (1-alpha)*variance + alpha*(value-mean)**2
        result.append(variance)
    return result


def projection(h, direction, alpha):
    if alpha == 1.0:
        return list(h)
    dot = sum(x*y for x, y in zip(h, direction))
    return [x-(1-alpha)*dot*y for x, y in zip(h, direction)]


def audit():
    checks = {}
    p, q = [0.8, 0.1, 0.1], [0.8, 0.199, 0.001]
    assert max(p) == max(q) and entropy(p) != entropy(q)
    checks['equal_max_probability_does_not_determine_entropy'] = {
        'max_probability': 0.8, 'entropy_p': entropy(p), 'entropy_q': entropy(q)}
    paper, public = paper_eat([1.0]*30), public_eat([1.0]*30)
    assert paper[0] > 0 and public[0] == 0 and paper[25] != public[25]
    checks['eat_paper_public_initialization_and_debiasing_differ'] = {
        'paper_value_at_observation_26': paper[25],
        'public_value_at_observation_26': public[25]}
    # A returned index i from public trace[1:] addresses observation i+1.
    values = [9.0]*26+[0.0]
    public_index = next(i for i, x in enumerate(values[1:]) if i >= 25 and x < 0.1)
    assert public_index == 25 and values[public_index] != values[public_index+1]
    checks['eat_public_sliced_index_needs_alignment_audit'] = {
        'returned_zero_based_index': public_index, 'matched_original_index': public_index+1}
    h, d, v, a, c = [2.0, 3.0], [1.0, 0.0], [1.0, 1.0], 0.7, -0.5
    add = lambda x, y: [i+j for i, j in zip(x, y)]
    cv = [c*x for x in v]
    after = projection(add(h, cv), d, a)
    before = add(projection(h, d, a), cv)
    assert after != before
    checks['projection_and_additive_rebalance_need_fixed_order'] = {
        'rebalance_then_projection': after, 'projection_then_rebalance': before,
        'predicted_difference': [-c*(1-a)*sum(x*y for x,y in zip(d,v))*z for z in d]}
    assert projection(h, d, 1.0) == h
    assert sum(x*x for x in projection(h,d,a)) <= sum(x*x for x in h)
    checks['projection_off_identity_and_local_norm_bound'] = True
    perpendicular = [0.0, c]
    assert projection(add(h,perpendicular),d,a) == add(projection(h,d,a),perpendicular)
    checks['orthogonality_removes_local_order_difference_only'] = True
    return {'scope': 'synthetic math and source-contract audit; not predictive or generation validation',
            'checks_passed': len(checks), 'checks': checks, 'gpu_used': False,
            'new_answers': 0, 'empirical_compression_claim': False}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
