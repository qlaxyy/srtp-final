"""New borrowed lexical mechanism, not an official CGRS implementation.

Fixed before new generation: subtract log(2) from reflection logits only at a
clean step opening, optionally requiring the existing ReBalance coefficient <0.
No probe, answer substitution, extra RNG, or claim that confidence proves truth.
"""
import math

PENALTY = math.log(2)
TRIGGERS = {
    14190: 'Wait', 13824: ' Wait', 11489: 'wait', 3783: ' wait',
    3983: 'But', 1988: ' But', 8088: 'but', 714: ' but',
    92014: 'Alternatively', 38478: ' Alternatively',
    75763: 'Alternative', 41109: ' Alternative',
    80022: 'Hmm', 88190: ' Hmm',
}


def token_flags(piece, rebalance_boundary):
    """Mixed boundary/content tokens must not open a lexical window."""
    opening = rebalance_boundary and not piece.rsplit('\n\n', 1)[-1].strip()
    return opening, not piece.strip()


def eligible(opening, thinking, coefficient, previous_mean, mode):
    if mode not in ('off', 'shadow', 'negative', 'always'):
        raise ValueError(mode)
    if mode == 'off' or not opening or not thinking:
        return False
    if mode == 'always':
        return True
    return math.isfinite(previous_mean) and math.isfinite(coefficient) and coefficient < 0


def next_opening(was_open, piece, boundary, token_id):
    if token_id in (151648, 151649):
        return False
    clean, whitespace = token_flags(piece, boundary)
    return clean if boundary else was_open and whitespace
