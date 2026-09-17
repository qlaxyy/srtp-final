"""CPU reference only; not installed in the inference runner.

Inputs are raw pre-temperature, pre-lexical-penalty logits from a ReBalance
forward. Auxiliary rows must have identical prefix/control snapshots. This
module deliberately does not pretend to implement temporary KV branches.
"""
import torch


def entropy(logits):
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1)


def cue_positions(last_positions, selected_rows, cue_lengths):
    """Decode-only positions, in explicitly selected active-batch row order."""
    if len(selected_rows) != len(cue_lengths):
        raise ValueError("one cue length per selected row required")
    if len(set(selected_rows)) != len(selected_rows):
        raise ValueError("duplicate rows")
    if any(n < 1 for n in cue_lengths):
        raise ValueError("empty cue")
    if any(i < 0 or i >= len(last_positions) for i in selected_rows):
        raise ValueError("row out of range")
    chunks = [last_positions[i] + torch.arange(1, n + 1,
              device=last_positions.device, dtype=last_positions.dtype)
              for i, n in zip(selected_rows, cue_lengths)]
    return torch.cat(chunks) if chunks else last_positions.new_empty(0)


def contrast(raw_logits, auxiliary_logits, selected_rows, *, enabled=False,
             scale=1.5):
    # Identity includes object identity: disabled mode must not normalize/cast.
    if not enabled or scale == 1 or not selected_rows:
        return raw_logits
    if len(set(selected_rows)) != len(selected_rows):
        raise ValueError("duplicate rows")
    if auxiliary_logits.shape != (len(selected_rows), raw_logits.shape[-1]):
        raise ValueError("auxiliary rows must match selected rows")
    if not torch.isfinite(raw_logits).all() or not torch.isfinite(auxiliary_logits).all():
        raise ValueError("contrast must precede sampling masks")
    result = raw_logits.float().clone()
    result[selected_rows] = (scale * raw_logits[selected_rows].float()
                             + (1 - scale) * auxiliary_logits.float())
    return result
