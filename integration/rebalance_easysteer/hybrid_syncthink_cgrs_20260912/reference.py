"""CPU specification oracle only. No vLLM hook, model import, or generation.

The proposed SyncThink-inspired adaptation follows equations (3)-(5), with
our explicit pacing cap 64. It is not an official SyncThink implementation.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SyncConfig:
    mode: str = "off"
    entropy_weight: float = 0.8
    pacing_cap: int = 64

    def __post_init__(self):
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError("Unknown mode")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("Invalid entropy weight")
        if type(self.pacing_cap) is not int or self.pacing_cap < 1:
            raise ValueError("Invalid pacing cap")


def raw_statistics(logits, end_id):
    """Natural-log entropy; zero-based descending-logit rank, ID breaks ties."""
    if len(logits) < 2 or not 0 <= end_id < len(logits):
        raise ValueError("Invalid vocabulary/end ID")
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("This oracle requires finite raw logits")
    maximum = max(logits)
    weights = [math.exp(v - maximum) for v in logits]
    total = math.fsum(weights)
    log_z_shift = math.log(total)
    entropy = math.fsum(w / total * (log_z_shift - (z - maximum))
                        for w, z in zip(weights, logits))
    end_logit = logits[end_id]
    rank = sum(z > end_logit or (z == end_logit and i < end_id)
               for i, z in enumerate(logits))
    return dict(entropy=entropy, rank=rank, max_probability=1 / total,
                end_probability=weights[end_id] / total)


def sync_action(logits, end_id, generated_count, in_think, config=SyncConfig()):
    """Return original object on off/shadow/non-trigger paths; never sample.

    generated_count counts accepted generated tokens, excluding prompt/replay.
    Caller must exclude partial prefill rows before calling this function.
    """
    if config.mode == "off":
        return logits, None
    if type(generated_count) is not int or generated_count < 0:
        raise ValueError("Invalid accepted-token count")
    if not in_think:
        return logits, None
    stats = raw_statistics(logits, end_id)
    t = generated_count + 1
    threshold = math.floor(min(t, config.pacing_cap)
                           * math.exp(-config.entropy_weight * stats['entropy']))
    stats.update(t=t, threshold=threshold, trigger=stats['rank'] <= threshold)
    if config.mode == "enforce" and stats['trigger']:
        result = [-math.inf] * len(logits)
        result[end_id] = 0.0
        return result, stats
    return logits, stats


def cgrs_probability(entropies, vocab_size, delta=0.9):
    """Paper equation, not public code's top-5 approximation."""
    if not 0 <= delta < 1 or vocab_size < 2 or not entropies:
        raise ValueError("Invalid probe/threshold")
    max_h = math.log(vocab_size)
    if any(not math.isfinite(h) or not 0 <= h <= max_h for h in entropies):
        raise ValueError("Invalid full-vocabulary entropy")
    certainty = 1 - math.fsum(entropies) / len(entropies) / max_h
    return certainty, max(0.0, (certainty - delta) / (1 - delta))


def token_interaction(means):
    """Negative additive difference means extra saving, units tokens/question."""
    if set(means) != {'U', 'R', 'S', 'RS'}:
        raise ValueError("All four factorial arms are necessary")
    return means['RS'] - means['R'] - means['S'] + means['U']
