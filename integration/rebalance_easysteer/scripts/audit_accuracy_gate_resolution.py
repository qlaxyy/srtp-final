"""Exact synthetic accuracy-gate calculations, never candidate efficacy estimates."""
import argparse
import itertools
import math
from pathlib import Path
import time

import numpy as np

from mechanism_candidates import require, save, sha


def distribution(n, discordance, delta):
    require(0 <= abs(delta) <= discordance <= 1, 'Invalid hypothetical probabilities')
    probabilities = np.array([(discordance-delta)/2, 1-discordance,
                              (discordance+delta)/2])
    result = np.array([1.])
    for _ in range(n):
        result = np.convolve(result, probabilities)
    require(abs(result.sum()-1) < 1e-12, 'Probability mass lost')
    return result


def binomial_reference(n, discordance, delta):
    if discordance == 0:
        return 1.0
    improve = (discordance+delta)/(2*discordance)
    # First draw the count of discordant questions; conditional improvements
    # follow a binomial law. Ties pass the observed nondecreasing-count rule.
    return sum(math.comb(n, d)*discordance**d*(1-discordance)**(n-d)*
               sum(math.comb(d, k)*improve**k*(1-improve)**(d-k)
                   for k in range((d+1)//2, d+1))
               for d in range(n+1))


def audit():
    started = time.perf_counter(); rows = []; checks = 0
    for n in (1, 2, 5):
        probabilities = [.07, .9, .03]
        exact = sum(math.prod(probabilities[value+1] for value in sequence)
                    for sequence in itertools.product((-1, 0, 1), repeat=n)
                    if sum(sequence) >= 0)
        computed = distribution(n, .1, -.04)[n:].sum()
        require(abs(exact-computed) < 1e-12, 'Exhaustive small case disagrees')
        checks += 1
    for q in (.05, .1, .2):
        for delta in (-.05, -.02, 0., .02):
            stages = {}
            for n in (100, 200):
                pmf = distribution(n, q, delta)
                probability = float(pmf[n:].sum())
                require(abs(probability-binomial_reference(n, q, delta)) < 1e-12,
                        'Conditional binomial and convolution disagree')
                expected = np.dot(np.arange(-n, n+1), pmf)
                require(abs(expected-n*delta) < 1e-10, 'Expected difference disagrees')
                stages[str(n)] = probability; checks += 2
            rows.append(dict(hypothetical_discordance=q, hypothetical_population_accuracy_delta_pp=100*delta,
                probability_observed_correct_count_not_lower=stages,
                probability_both_independent_accuracy_gates_pass=stages['100']*stages['200']))
    return dict(status='completed_exact_synthetic_accuracy_gate_resolution_not_model_results',
        derivation='D_i in{-1,0,+1}; P(-1)=(q-delta)/2, P(0)=1-q, P(+1)=(q+delta)/2. Sum of n independent differences has the n-fold convolution. The observed accuracy gate is sum>=0. Independent conditional-binomial enumeration checks it.',
        scenarios=rows, independent_checks=checks, cpu_seconds=time.perf_counter()-started,
        zero_observed_discordance='A nonparametric paired bootstrap then always returns accuracy delta0 and a degenerate[0,0] interval; it cannot generate unseen discordant outcomes or establish zero population loss.',
        decision='Keep the fixed100-question screen and conditional200-question reserve. Report uncertainty and limits; do not change the gate, add repetitions, or inflate confirmation claims.',
        limits='IID hypothetical problem outcomes, not measured population rates or a forecast of these candidates. Token/cap gates and question/seed dependence are not modeled. The product assumes independent fresh stages and is not the false-promotion probability of the full research procedure.',
        source_sha256=sha(Path(__file__), source=True), model_loads=0, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable audit exists')
    result = audit(); save(args.output, result)
    print({key:result[key] for key in ('status', 'independent_checks', 'cpu_seconds')})
    print(next(row for row in result['scenarios'] if row['hypothetical_discordance']==.1
               and row['hypothetical_population_accuracy_delta_pp']==-2))


if __name__ == '__main__':
    main()
