"""C-probe v1: explicit paper-derived policy, without model dependencies."""
from dataclasses import dataclass, field
import hashlib
import math
import unicodedata

TRIGGERS = {
    14190: 'Wait', 13824: ' Wait', 11489: 'wait', 3783: ' wait',
    3983: 'But', 1988: ' But', 8088: 'but', 714: ' but',
    92014: 'Alternatively', 38478: ' Alternatively',
    75763: 'Alternative', 41109: ' Alternative',
    80022: 'Hmm', 88190: ' Hmm',
}
PROBE_PROMPT = '\n**Final Answer**\n\\boxed'


def problem_hash(problem):
    normalized = ''.join(unicodedata.normalize('NFKC', problem).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    interval: int = 1024
    max_probes: int = 8
    probe_tokens: int = 32
    delta: float = .9
    max_tokens: int = 16000
    seed: int = 42

    def __post_init__(self):
        if (self.interval < 1 or self.max_probes < 1 or self.probe_tokens < 2
                or self.max_tokens < 2 or not 0 <= self.delta < 1):
            raise ValueError('Invalid C-probe configuration')


def probability(certainty, delta=.9):
    if not math.isfinite(certainty) or not 0 <= certainty <= 1:
        raise ValueError('Certainty outside [0,1]')
    if not 0 <= delta < 1:
        raise ValueError('delta must be below one')
    return max(0., (certainty - delta) / (1. - delta))


def draw(key, position, seed):
    """Counter RNG separate from all model sampling RNGs and batch ordering."""
    payload = f'cgrs-probe-v1|{key}|{position}|{seed}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / 2**64


def boxed_certainty(token_ids, entropies, tokenizer, vocab_size):
    """Only full tokens strictly inside the first complete nonempty box.

    Tokens crossing a brace are excluded, not partially counted. Invalid or
    unfinished probes yield zero; no stale confidence is retained.
    """
    if len(token_ids) != len(entropies) or vocab_size < 2:
        raise ValueError('Probe entropy alignment mismatch')
    prefixes = [tokenizer.decode(token_ids[:i], skip_special_tokens=False)
                for i in range(len(token_ids) + 1)]
    if any(not b.startswith(a) for a, b in zip(prefixes, prefixes[1:])):
        return 0., 'non_monotonic_decode', []
    text = prefixes[-1]
    start = text.find('{')
    if start < 0 or text[:start].strip():
        return 0., 'missing_box', []
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] in '{}':
            # Ignore braces escaped by an odd number of backslashes.
            backslashes, j = 0, i - 1
            while j >= 0 and text[j] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2:
                continue
            depth += 1 if text[i] == '{' else -1
            if depth == 0:
                end = i
                break
    if end is None or not text[start + 1:end].strip():
        return 0., 'incomplete_or_empty_box', []
    selected = [i for i in range(len(token_ids))
                if len(prefixes[i]) > start and len(prefixes[i + 1]) <= end
                and prefixes[i + 1][len(prefixes[i]):].strip()]
    if not selected:
        return 0., 'no_interior_tokens', []
    values = [entropies[i] for i in selected]
    if any(not math.isfinite(h) or h < 0 or h > math.log(vocab_size) + 1e-5
           for h in values):
        raise ValueError('Invalid full-vocabulary entropy')
    return max(0., min(1., 1 - sum(values) / len(values)
                       / math.log(vocab_size))), 'complete', selected


@dataclass
class State:
    key: str
    config: Config
    count: int = 0
    thinking: bool = True
    opening: bool = False
    p: float = 0.
    probes: int = 0
    last_probe: int = 0
    probe_output_tokens: int = 0
    probe_prompt_tokens: int = 0
    mask_count: int = 0
    mask_positions: list = field(default_factory=list)

    @property
    def remaining(self):
        return (self.config.max_tokens - self.count
                - self.probe_output_tokens - self.probe_prompt_tokens)

    def accept(self, token, piece, boundary_ids):
        self.count += 1
        if token == 151649:
            self.thinking, self.opening, self.p = False, False, 0.
        elif self.thinking and token in boundary_ids:
            self.p = 0.
            # R's original token boundary remains unchanged. A token already
            # containing next-step content is not a lexical opening for C.
            self.opening = not piece.rsplit('\n\n', 1)[-1].strip()
        elif piece.strip():
            self.opening = False
        return (self.thinking and self.opening and token in boundary_ids
                and self.count - self.last_probe >= self.config.interval
                and self.probes < self.config.max_probes)

    def reserve_probe(self, prompt_tokens):
        # Reserve at least one subsequent main-chain output token.
        if self.remaining < prompt_tokens + self.config.probe_tokens + 1:
            self.p = 0.
            return False
        self.probes += 1
        self.last_probe = self.count
        self.probe_prompt_tokens += prompt_tokens
        return True

    def complete_probe(self, certainty, output_tokens):
        if not 0 <= output_tokens <= self.config.probe_tokens:
            raise ValueError('Probe output exceeds reservation')
        self.probe_output_tokens += output_tokens
        self.p = probability(certainty, self.config.delta)

    def should_mask(self):
        return (self.thinking and self.opening and self.p > 0
                and draw(self.key, self.count, self.config.seed) < self.p)
