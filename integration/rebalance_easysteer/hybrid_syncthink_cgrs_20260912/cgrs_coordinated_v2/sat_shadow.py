"""Bounded raw-logit recorder around RC14, with no intervention or CPU sync.

Install after Adapter; close before Adapter.close(). For engineering only:
record the complete <=512-token batch, then copy to CPU after generation.
No preemption, new request waves, asynchronous scheduling, or slot reuse.
"""


class ShadowRecorder:
    def __init__(self, runner, *, enabled=False, max_calls=512):
        self.enabled = enabled
        if not enabled:
            return
        if not 1 <= max_calls <= 512:
            raise ValueError('Bounded engineering capture only')
        self.runner, self.original = runner, runner.sampler
        self.max_calls, self.frames = max_calls, []
        runner.sampler = self

    def __getattr__(self, name):
        original = self.__dict__.get('original')
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    def __call__(self, logits, batch, **kwargs):
        import torch
        if len(self.frames) >= self.max_calls:
            raise RuntimeError('Capture limit; retain partials, no silent truncation')
        if batch.num_draft_tokens or logits.shape[0] != batch.num_reqs or batch.num_reqs > 8:
            raise ValueError('At most eight single-token requests required')
        idx = batch.idx_mapping[:batch.num_reqs].long().clone()
        valid = torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens >= batch.prefill_len_np,
                                device=logits.device).clone()
        # All gather tensors are separate storage; inner RC14 modifies logits.
        values, ids = torch.topk(logits, 512, dim=-1, sorted=True)
        values = values.float()
        result = self.original(logits, batch, **kwargs)
        if result.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise ValueError('Unsupported token shape')
        self.frames.append((idx, valid, values, ids, result.sampled_token_ids[:, 0].clone()))
        return result

    def export(self):
        """Call after synchronized generation. Return accepted tokens per slot."""
        rows = {}
        for frame in self.frames:
            idx, valid, values, ids, selected = [x.detach().cpu().numpy() for x in frame]
            for i, slot in enumerate(idx):
                if valid[i]:
                    rows.setdefault(int(slot), []).append(dict(values=values[i], ids=ids[i], selected=int(selected[i])))
        return rows

    def close(self):
        if self.enabled:
            if self.runner.sampler is not self:
                raise RuntimeError('Close recorder before underlying adapter')
            self.runner.sampler = self.original
