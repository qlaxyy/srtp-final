"""Instance instrumentation around ORIGINAL L27; forced saved tokens, no policy fit."""
import torch


class FeedbackCapture:
    def __init__(self, adapter, cap):
        self.owner = adapter
        self.runner = adapter.runner
        n, device = self.runner.max_num_reqs, self.runner.device
        self.targets = torch.zeros((n, cap), dtype=torch.long, device=device)
        self.length = torch.zeros(n, dtype=torch.long, device=device)
        self.traces = {k: torch.full((n, cap), float('nan'), device=device)
                       for k in ['rawmax', 'logp', 'coefficient_before', 'lex_state', 'lex_gate']}
        self.native = adapter.original_sampler
        self.l27 = self.runner.sampler
        self.last_raw = None

    def register(self, slot, ids):
        if not ids or len(ids) > self.targets.shape[1]:
            raise ValueError('Invalid complete saved response')
        self.targets[slot].zero_()
        self.targets[slot, :len(ids)] = torch.as_tensor(ids, device=self.targets.device)
        self.length[slot] = len(ids)
        for values in self.traces.values():
            values[slot].fill_(float('nan'))

    def outer(self, logits, batch, **kwargs):
        # Exact expression used by native model_runner before sampler mutation.
        raw = logits.float()
        self.last_raw = torch.exp(raw.amax(-1)-torch.logsumexp(raw, -1))
        return self.l27(logits, batch, **kwargs)

    def install(self):
        self.owner.original_sampler = self
        self.runner.sampler = self.outer

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, logits, batch, **kwargs):
        # Called INSIDE original L27 sampler, after penalty but before lexical
        # acceptance or ReBalance observation. No logits mutation here.
        o = self.owner
        idx = batch.idx_mapping[:batch.num_reqs].long()
        valid = torch.as_tensor(batch.num_computed_tokens_np + batch.num_scheduled_tokens
                                >= batch.prefill_len_np, device=logits.device)
        if self.native.returns_logprobs(batch.idx_mapping_np):
            raise ValueError('Native sampled-token logprobs invalid for forced tokens')
        pos = o.count[idx]
        torch._assert_async((~valid | (pos < self.length[idx])).all(), 'Saved response exhausted')
        safe = pos.clamp(max=self.targets.shape[1]-1)
        target = self.targets[idx, safe]
        z = logits.float()
        lp = z.gather(1, target[:, None]).squeeze(1)-torch.logsumexp(z, -1)
        state = self.runner.steer_vector_state
        gate = valid & o.lex_open[idx] & o.thinking[idx]
        gate &= torch.isfinite(state._coefs[idx]) & torch.isfinite(state._prev_step_mean[idx]) & (state._coefs[idx] < 0)
        fields = dict(rawmax=self.last_raw, logp=lp,
            coefficient_before=state._coefs[idx]*state._in_think[idx],
            lex_state=o.lex_state[idx].float(), lex_gate=gate.float())
        for key, value in fields.items():
            self.traces[key][idx, safe] = torch.where(valid, value, self.traces[key][idx, safe])
        output = self.native(logits, batch, **kwargs)
        if output.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise ValueError('Sampler ABI changed')
        output.sampled_token_ids[:, 0] = torch.where(valid, target.to(output.sampled_token_ids.dtype), output.sampled_token_ids[:, 0])
        return output

    def completed(self, slot):
        length = int(self.length[slot])
        if int(self.owner.count[slot]) != length:
            raise ValueError('Incomplete forced response')
        result = {k: x[slot, :length].detach().cpu().numpy().copy() for k, x in self.traces.items()}
        if not all(__import__('numpy').isfinite(x).all() for x in result.values()):
            raise ValueError('Incomplete/invalid scoring trace')
        return result

    def close(self):
        self.runner.sampler = self.l27
        self.owner.original_sampler = self.native
