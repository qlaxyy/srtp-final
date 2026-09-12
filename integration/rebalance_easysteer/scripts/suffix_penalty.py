"""Bounded CPU reference for the fixed long-suffix sampling candidate."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SuffixPenaltyConfig:
    allowed_length: int = 16
    history_tokens: int = 1024
    multiplier: float = .8
    base: float = 1.75
    maximum_penalty: float = 20.

    def __post_init__(self):
        for value in [self.allowed_length, self.history_tokens]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Invalid suffix window/threshold')
        if self.allowed_length >= self.history_tokens:
            raise ValueError('Suffix threshold must fit history with an earlier witness')
        if not all(math.isfinite(x) for x in [self.multiplier, self.base, self.maximum_penalty]):
            raise ValueError('Nonfinite penalty configuration')
        if not 0 < self.multiplier <= self.maximum_penalty or self.base < 1:
            raise ValueError('Invalid penalty scale')

    def penalty(self, length):
        if length < self.allowed_length: return 0.
        exponent = (length-self.allowed_length)*math.log(self.base)
        cap = math.log(self.maximum_penalty/self.multiplier)
        return min(self.maximum_penalty, self.multiplier*math.exp(min(exponent, cap)))


class SuffixMonitor:
    """Ring-buffer DP; every feed sees one already generated token only."""
    def __init__(self, breakers, config=SuffixPenaltyConfig()):
        self.config = config; self.breakers = set(breakers)
        self.tokens = np.full(config.history_tokens, -1, dtype=np.int64)
        self.matches = np.zeros(config.history_tokens, dtype=np.int32)
        self.positions = np.full(config.history_tokens, -1, dtype=np.int64)
        self.slot = np.arange(config.history_tokens); self.arrived = 0

    def feed(self, token):
        if not isinstance(token, (int, np.integer)) or token < 0:
            raise ValueError('Invalid token ID')
        width = self.config.history_tokens; position = self.arrived; slot = position % width
        self.tokens[slot] = token; self.positions[slot] = position
        if token in self.breakers:
            self.matches.fill(0)
        else:
            new = (np.roll(self.matches, 1)+1)*(self.tokens == token)
            limit = self.slot+1 if position < width else (self.slot-slot-1) % width+1
            self.matches = np.minimum(new, limit).astype(np.int32)
            self.matches[slot] = 0
        self.arrived += 1

    def candidates(self):
        """Return next ID -> (longest match, earlier ending position)."""
        result = {}; width = self.config.history_tokens
        for slot in np.flatnonzero(self.matches >= self.config.allowed_length):
            next_token = int(self.tokens[(slot+1) % width])
            if next_token < 0 or next_token in self.breakers: continue
            value = (int(self.matches[slot]), int(self.positions[slot]))
            old = result.get(next_token)
            if old is None or value[0] > old[0] or (value[0] == old[0] and value[1] < old[1]):
                result[next_token] = value
        return result

    def saved_next_token_witness(self, token):
        """Offline eligibility of one saved next token; does not change state."""
        if token in self.breakers: return None
        mask = (self.matches >= self.config.allowed_length) & (np.roll(self.tokens, -1) == token)
        if not mask.any(): return None
        length = int(self.matches[mask].max())
        return length, int(self.positions[mask & (self.matches == length)].min())


def brute_force(ids, breakers, config):
    """Independent explicit backward search, including overlapping witnesses."""
    offset = max(0, len(ids)-config.history_tokens); window = ids[offset:]
    if not window or window[-1] in breakers: return {}
    result = {}; last = len(window)-1
    for i in range(last):
        if window[i] != window[last] or window[i+1] in breakers: continue
        n = 1
        while i-n >= 0 and window[i-n] == window[last-n] and window[last-n] not in breakers:
            n += 1
        if n < config.allowed_length: continue
        token = window[i+1]; value = (n, offset+i)
        old = result.get(token)
        if old is None or n > old[0] or (n == old[0] and value[1] < old[1]): result[token] = value
    return result


def checks():
    generator = np.random.default_rng(20260925); comparisons = 0
    configs = [SuffixPenaltyConfig(allowed_length=length, history_tokens=window)
               for window in [3, 7, 17] for length in [1, 2]]
    streams = [[1]*64, [1, 2, 3]*40, [1, 2, 9, 1, 2, 3]*20,
               list(range(80)), [1, 2, 3, 4, 5, 6, 7, 8]*20]
    streams += [generator.integers(0, 10, size=180).tolist() for _ in range(12)]
    for config in configs:
        for ids in streams:
            monitor = SuffixMonitor({0, 9}, config); prefix = []
            for token in ids:
                if monitor.candidates() != brute_force(prefix, {0, 9}, config):
                    raise AssertionError(f'Suffix reference disagrees at{config}/{prefix}')
                assert monitor.saved_next_token_witness(token) == monitor.candidates().get(token)
                prefix.append(token); monitor.feed(token); comparisons += 1
            assert monitor.tokens.size == config.history_tokens
    standard = SuffixPenaltyConfig()
    long_stream = [1, 2, 3]*800; monitor = SuffixMonitor({9}, standard)
    for i, token in enumerate(long_stream):
        if i in [0, 15, 16, 17, 128, 1023, 1024, 1025, 2048]:
            assert monitor.candidates() == brute_force(long_stream[:i], {9}, standard)
            comparisons += 1
        monitor.feed(token)
    assert standard.penalty(15) == 0. and standard.penalty(16) == .8
    assert standard.penalty(100000) == 20. and all(math.isfinite(standard.penalty(i)) for i in range(16000))
    return dict(scalar_reference_prefixes=comparisons, bounded_buffers=True, finite_penalties=True,
                overlapping_matches=True, breakers=True, expired_history=True)


if __name__ == '__main__': print(checks())
