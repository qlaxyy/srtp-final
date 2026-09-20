"""Fixed-token replay at the V2 sampler boundary; never alter raw logits.

Experimental instrumentation, not a replacement ReBalance controller. The
runner computes raw confidence before this call and observes forced tokens
after it. Caller must reject preemption, speculation and probability outputs.
"""
from __future__ import annotations

import torch


class ReplayBuffers:
    def __init__(self, slots, cap, device):
        self.targets = torch.zeros((slots, cap), dtype=torch.long, device=device)
        self.logmax = torch.full((slots, cap), float('nan'), device=device)
        self.count = torch.zeros(slots, dtype=torch.long, device=device)
        self.length = torch.zeros_like(self.count)
        self.prompt = torch.zeros_like(self.count)

    def register(self, slot, tokens, prompt_length):
        if not tokens or len(tokens) > self.targets.shape[1]:
            raise ValueError('Invalid saved trajectory length')
        self.targets[slot].zero_()
        self.targets[slot, :len(tokens)] = torch.as_tensor(tokens, device=self.targets.device)
        self.logmax[slot].fill_(float('nan'))
        self.count[slot] = 0
        self.length[slot] = len(tokens)
        self.prompt[slot] = prompt_length

    def capture(self, logits, idx, valid, seq_lens):
        pos = self.count[idx]
        torch._assert_async((~valid | (seq_lens == self.prompt[idx] + pos)).all(),
                            'Replay prefix alignment mismatch')
        torch._assert_async((~valid | (pos < self.length[idx])).all(),
                            'Replay exceeded saved trajectory')
        raw = logits.float()
        lp = raw.amax(-1) - torch.logsumexp(raw, dim=-1)
        torch._assert_async((~valid | torch.isfinite(lp)).all(), 'Invalid raw probability')
        safe_pos = pos.clamp(max=self.targets.shape[1]-1)
        target = self.targets[idx, safe_pos]
        self.logmax[idx, safe_pos] = torch.where(valid, lp, self.logmax[idx, safe_pos])
        self.count[idx] += valid.long()
        return target

    def completed(self, slot):
        count, length = int(self.count[slot]), int(self.length[slot])
        if count != length:
            raise RuntimeError(f'Incomplete replay: {count}/{length}')
        values = self.logmax[slot, :length].cpu().tolist()
        if not all(__import__('math').isfinite(x) for x in values):
            raise RuntimeError('Missing replay probability')
        return values


class MotivationReplaySampler:
    def __init__(self, original, buffers, force=True):
        self.original, self.buffers = original, buffers
        self.force = force

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, logits, batch, **kwargs):
        if batch.num_draft_tokens or logits.shape[0] != batch.num_reqs:
            raise RuntimeError('Replay requires one prediction per request')
        idx = batch.idx_mapping[:batch.num_reqs].long()
        if self.original.returns_logprobs(batch.idx_mapping_np):
            raise RuntimeError('Native sampled-token logprobs cannot describe forced tokens')
        valid = torch.as_tensor(batch.num_computed_tokens_np + batch.num_scheduled_tokens
                                >= batch.prefill_len_np, device=logits.device)
        forced = self.buffers.capture(logits, idx, valid, batch.seq_lens[:batch.num_reqs])
        output = self.original(logits, batch, **kwargs)
        if output.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise RuntimeError('Unexpected sampler ABI')
        # Preserve native prefill/count metadata; change only accepted token IDs.
        if self.force:
            output.sampled_token_ids[:, 0] = torch.where(
                valid, forced.to(output.sampled_token_ids.dtype), output.sampled_token_ids[:, 0])
        return output
