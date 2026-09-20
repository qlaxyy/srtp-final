"""Differentiable HF surrogate with recorded native history and exact token mask.

Not the online vLLM policy: coefficients and lexical gates are frozen here.
No model/controller copy, fitting loop, or automatic deployment.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint


def effective_scales(input_ids, history, prompt_length, boundary_ids):
    if input_ids.ndim != 1 or history.shape != input_ids.shape:
        raise ValueError('Input/history shape mismatch')
    if not 0 < prompt_length <= len(input_ids):
        raise ValueError('Invalid prompt length')
    boundary = torch.isin(input_ids, boundary_ids)
    generated = torch.arange(len(input_ids), device=input_ids.device) >= prompt_length
    return history * (boundary & generated)


class FixedHistoryScorer:
    def __init__(self, model, original_vector, boundaries, tables, layer=20, chunk=128):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Base model must be frozen')
        self.model, self.vector, self.boundaries, self.tables = model, original_vector, boundaries, tables
        self.chunk = chunk
        self.scales, self.delta = None, None
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        z = output[0] if isinstance(output, tuple) else output
        if z.shape[:2] != (1, len(self.scales)):
            raise ValueError('Only full single-trajectory replay is supported')
        alpha = self.scales.to(z.dtype)[None, :, None]
        steered = z + alpha * self.vector.to(z.dtype)[None, None, :]
        if self.delta is not None:
            steered = steered + (-alpha).clamp_min(0) * self.delta.to(z.dtype)[None, None, :]
        return (steered,)+output[1:] if isinstance(output, tuple) else steered

    def score(self, prompt, response, trace, delta=None):
        device = self.vector.device
        if response[-1] != 151643:
            raise ValueError('Full terminal supervision required')
        ids = torch.tensor(prompt + response, device=device)
        hist = torch.as_tensor(trace['history'], device=device)
        if len(hist) != len(ids) or len(trace['lex_state']) != len(response):
            raise ValueError('Trace/sequence alignment mismatch')
        self.scales = effective_scales(ids[:-1], hist[:-1], len(prompt), self.boundaries)
        self.delta = delta
        h = self.model.model(input_ids=ids[:-1][None, :], use_cache=False, return_dict=True).last_hidden_state[0]
        h = h[len(prompt)-1:]
        targets = ids[len(prompt):]
        states = torch.as_tensor(trace['lex_state'], device=device, dtype=torch.long)
        gates = torch.as_tensor(trace['lex_gate'], device=device, dtype=torch.bool)
        candidates = self.tables['candidate_ids']
        hits = self.tables['hits'][:, self.tables['token_classes'][candidates].long()]
        def chunk_score(hidden, target, state, gate):
            logits = self.model.lm_head(hidden).float()
            lexical = hits[state] & gate[:, None]
            # Same fixed L27 subtraction. No sampling top-p or temperature.
            values = logits[:, candidates]
            logits[:, candidates] = torch.where(lexical, values-math.log(2), values)
            return logits.gather(1, target[:, None]).squeeze(1)-torch.logsumexp(logits, -1)
        result = []
        for i in range(0, len(response), self.chunk):
            end = i+self.chunk
            args = (h[i:end], targets[i:end], states[i:end], gates[i:end])
            if delta is not None and torch.is_grad_enabled():
                # Recompute vocabulary projection during backward instead of
                # retaining N * vocabulary-size logits for all chunks.
                part = checkpoint(chunk_score, *args, use_reentrant=False)
            else:
                part = chunk_score(*args)
            result.append(part)
        return torch.cat(result)

    def close(self):
        self.handle.remove()
