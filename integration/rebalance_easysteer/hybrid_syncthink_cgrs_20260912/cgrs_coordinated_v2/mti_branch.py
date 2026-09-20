"""Isolated HF cache reference and paged-copy plan, not a vLLM adapter.

Deep-copying the HF cache is intentional for the engineering oracle. It is
not the proposed high-throughput implementation and must not be benchmarked
as the production combination.
"""
import copy
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ScratchPlan:
    shared_full_blocks: int
    copy_tail_tokens: int
    scratch_blocks: int
    cue_positions: tuple


def scratch_plan(prefix_length, cue_length, block_size):
    if prefix_length < 1 or cue_length < 1 or block_size < 1:
        raise ValueError('positive lengths required')
    full, tail = divmod(prefix_length, block_size)
    return ScratchPlan(full, tail, (tail + cue_length + block_size - 1) // block_size,
                       tuple(range(prefix_length, prefix_length + cue_length)))


def isolated_cue(decoder, cache, prefix_length, cue_ids):
    """One unpadded request; decoder must be eval-only and have no state hooks.

    No sampler, controller, accepted-token list, or lexical state is passed in.
    Output owns its branch tensors; the copied cache is discarded on return.
    """
    if cue_ids.ndim != 2 or cue_ids.shape[0] != 1 or cue_ids.shape[1] == 0:
        raise ValueError('one nonempty unpadded cue required')
    if prefix_length < 1 or cache.get_seq_length() != prefix_length:
        raise ValueError('prefix/cache clock mismatch')
    if getattr(decoder, 'training', True):
        raise ValueError('eval decoder required')
    devices = [cue_ids.device.index] if cue_ids.is_cuda else []
    branch = copy.deepcopy(cache)
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            length = cue_ids.shape[1]
            positions = torch.arange(prefix_length, prefix_length + length,
                                     device=cue_ids.device).unsqueeze(0)
            result = decoder(input_ids=cue_ids, past_key_values=branch,
                             position_ids=positions,
                             attention_mask=torch.ones((1, prefix_length + length),
                                                       device=cue_ids.device, dtype=torch.long),
                             use_cache=True, return_dict=True)
            return result.last_hidden_state[:, -1:, :].clone()
    finally:
        del branch
