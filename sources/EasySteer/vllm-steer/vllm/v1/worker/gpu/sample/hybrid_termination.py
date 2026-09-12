# SPDX-License-Identifier: Apache-2.0
"""Opt-in SyncThink-inspired termination; no changes to ReBalance statistics."""

import torch

NAMESPACE = "hybrid_syncthink_cgrs_20260912"


def admit(runner, request, slot):
    params = request.sampling_params
    config = (params.extra_args or {}).get(NAMESPACE) if params else None
    if not config or config.get("mode", "off") == "off":
        return
    if (params.n != 1 or params.min_tokens or params.logit_bias
            or params.bad_words or params.presence_penalty
            or params.frequency_penalty or params.repetition_penalty != 1
            or params.structured_outputs is not None
            or runner.speculative_config is not None
            or runner.parallel_config.tensor_parallel_size != 1
            or runner.cache_config.enable_prefix_caching):
        raise ValueError("S64 does not support these request/engine settings")
    state = getattr(runner, "hybrid_termination", None)
    if state is None:
        state = HybridTerminationState(runner.max_num_reqs, runner.device)
        runner.hybrid_termination = state
    state.add_request(request.req_id, slot, request.prompt_token_ids,
                      len(request.prefill_token_ids), config)


class HybridTerminationState:
    """Accepted-token clocks owned by live request IDs, independent of R."""

    def __init__(self, capacity, device):
        self.device = device
        self.requests = {}
        self.completed = {}
        self.count = torch.zeros(capacity, dtype=torch.long, device=device)
        self.closed = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.first_trigger = torch.full_like(self.count, -1)
        self.end_position = torch.full_like(self.count, -1)
        self.trigger_count = torch.zeros_like(self.count)
        self.first_rank = torch.full_like(self.count, -1)
        self.first_entropy = torch.zeros(capacity, device=device)
        self.events = []
        self.max_buffer_estimate = 0

    def add_request(self, req_id, slot, prompt, prefill_length, config):
        if config != {"mode": config.get("mode"), "entropy_weight": 0.8,
                      "pacing_cap": 64, "end_token_id": 151649}:
            raise ValueError("S64 requires the fixed registered configuration")
        if config["mode"] not in ("shadow", "enforce"):
            raise ValueError("Unknown hybrid mode")
        if prefill_length != len(prompt) or 151648 not in prompt:
            raise ValueError("S64 requires a fresh thinking request")
        if req_id in self.requests or req_id in self.completed:
            raise ValueError("Request identity reused")
        self.requests[req_id] = (slot, len(prompt), config["mode"])
        self.count[slot] = 0
        self.closed[slot] = False
        self.first_trigger[slot] = self.end_position[slot] = -1
        self.trigger_count[slot] = 0
        self.first_rank[slot] = -1
        self.first_entropy[slot] = 0

    def remove_request(self, req_id):
        data = self.requests.pop(req_id, None)
        if data is None:
            return
        slot, prompt_len, mode = data
        values = torch.stack([self.count[slot], self.first_trigger[slot],
                              self.end_position[slot], self.trigger_count[slot],
                              self.first_rank[slot]]).cpu().tolist()
        self.completed[req_id] = dict(
            accepted_tokens=values[0], first_trigger=values[1],
            end_position=values[2], trigger_count=values[3],
            first_rank=values[4], first_entropy=float(self.first_entropy[slot]),
            prompt_length=prompt_len, mode=mode)

    def read_apply(self, logits, batch):
        if batch.num_draft_tokens or logits.shape[0] != batch.num_reqs:
            raise ValueError("S64 requires one logits row per request")
        positions = [p for p, rid in enumerate(batch.req_ids)
                     if rid in self.requests and
                     batch.num_computed_tokens_np[p]
                     + batch.num_scheduled_tokens[p] >= batch.prefill_len_np[p]]
        if not positions:
            return None
        if logits.shape[1] <= 151649:
            raise ValueError("Wrong vocabulary")
        start = end = None
        if logits.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        pos = torch.tensor(positions, device=logits.device)
        idx = batch.idx_mapping.index_select(0, pos).long()
        prompt_lengths = torch.tensor(
            [self.requests[batch.req_ids[p]][1] for p in positions],
            device=logits.device)
        torch._assert_async((batch.seq_lens[pos] == prompt_lengths
                             + self.count[idx]).all(),
                            "Accepted-token clock or replay mismatch")
        f = logits.index_select(0, pos).float()
        torch._assert_async((~torch.isnan(f) & ~torch.isposinf(f)).all(),
                            "Invalid raw logits")
        log_z = torch.logsumexp(f, dim=-1)
        torch._assert_async(torch.isfinite(log_z).all(), "Empty support")
        prob = torch.exp(f - log_z[:, None])
        entropy = (log_z - (prob * f.masked_fill(
            ~torch.isfinite(f), 0)).sum(-1)).clamp_min(0)
        end_logit = f[:, 151649:151650]
        ids = torch.arange(f.shape[1], device=f.device)
        rank = ((f > end_logit) | ((f == end_logit) & (ids < 151649))).sum(-1)
        count = self.count[idx]
        threshold = torch.floor((count + 1).clamp_max(64).float()
                                * torch.exp(-0.8 * entropy))
        trigger = (rank <= threshold) & ~self.closed[idx]
        first = trigger & (self.first_trigger[idx] < 0)
        self.first_trigger[idx] = torch.where(first, count, self.first_trigger[idx])
        self.first_rank[idx] = torch.where(first, rank, self.first_rank[idx])
        self.first_entropy[idx] = torch.where(first, entropy, self.first_entropy[idx])
        self.trigger_count[idx] += trigger.long()
        enforce = torch.tensor([self.requests[batch.req_ids[p]][2] == "enforce"
                                for p in positions], device=f.device)
        force = trigger & enforce
        replacement = torch.full((f.shape[1],), -torch.inf,
                                 device=logits.device, dtype=logits.dtype)
        replacement[151649] = 0
        # Shadow returns the original logits without any copy/rounding.
        if any(self.requests[batch.req_ids[p]][2] == "enforce" for p in positions):
            logits[pos] = torch.where(force[:, None], replacement, logits[pos])
        self.max_buffer_estimate = max(self.max_buffer_estimate,
                                       f.numel() * 4 * 4)
        if end is not None:
            end.record()
            self.events.append((start, end))
        return pos, idx

    def observe(self, ticket, sampled):
        if ticket is None:
            return
        pos, idx = ticket
        tokens = sampled[:, 0].index_select(0, pos)
        closed_now = (tokens == 151649) & ~self.closed[idx]
        self.end_position[idx] = torch.where(
            closed_now, self.count[idx], self.end_position[idx])
        self.closed[idx] |= closed_now
        self.count[idx] += 1

    def finish_all(self):
        for rid in list(self.requests):
            self.remove_request(rid)
        if self.events:
            torch.cuda.synchronize()
        return dict(requests=self.completed,
                    statistics_and_mask_gpu_ms=sum(a.elapsed_time(b)
                                                   for a, b in self.events),
                    buffer_bytes_estimate=self.max_buffer_estimate,
                    extra_model_forwards=0, probe_tokens=0)
