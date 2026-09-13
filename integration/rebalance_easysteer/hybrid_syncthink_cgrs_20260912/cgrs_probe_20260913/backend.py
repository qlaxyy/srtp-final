"""Opt-in instance adapter for the pinned EasySteer V2 runner.

No production files are patched. Main requests retain KV/controller/RNG state;
probe requests replay exact token IDs and the recorded historical R scales.
This implementation intentionally rejects asynchronous/multiprocess engines.
"""
from contextlib import contextmanager
import hashlib
import time

from policy import TRIGGERS, boxed_certainty


def enable_cumulative(llm, request_ids, cumulative_kind=None):
    """Undo offline FINAL_ONLY in copied output states before any execution."""
    if cumulative_kind is None:
        from vllm.sampling_params import RequestOutputKind
        cumulative_kind = RequestOutputKind.CUMULATIVE
    engine = llm.llm_engine
    requests = engine.engine_core.engine_core.scheduler.requests
    states = engine.output_processor.request_states
    for rid in request_ids:
        if requests[rid].num_output_tokens:
            raise RuntimeError('Configure streaming before the first generated token')
        states[rid].output_kind = cumulative_kind
        states[rid].stream_interval = 1
        requests[rid].sampling_params.output_kind = cumulative_kind


@contextmanager
def parked(scheduler, allow_waiting=False):
    """Temporarily withhold live main requests; retain their KV ownership."""
    if not allow_waiting and (scheduler.waiting or scheduler.skipped_waiting):
        raise RuntimeError('Park only after all primary requests are admitted')
    queues = None
    if allow_waiting:
        queues = scheduler.waiting, scheduler.skipped_waiting
        scheduler.waiting = type(queues[0])()
        scheduler.skipped_waiting = type(queues[1])()
    held = scheduler.running
    scheduler.running = []
    try:
        yield held
        if scheduler.running or scheduler.waiting or scheduler.skipped_waiting:
            raise RuntimeError('Probe requests have not drained')
    finally:
        # On exceptions retain all live requests so abort/cleanup can find them.
        scheduler.running = held + scheduler.running
        if queues is not None:
            for saved, active in zip(queues, (scheduler.waiting, scheduler.skipped_waiting)):
                for request in active:
                    saved.add_request(request)
            scheduler.waiting, scheduler.skipped_waiting = queues


class SamplerAdapter:
    def __init__(self, owner, original):
        self.owner, self.original = owner, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, logits, batch, **kwargs):
        owner = self.owner
        torch = owner.torch
        start = time.perf_counter()
        if batch.num_draft_tokens or logits.shape[0] != batch.num_reqs:
            raise RuntimeError('C-probe requires one prediction per request')
        counted = set()
        for pos, rid in enumerate(batch.req_ids):
            probe = owner.probes.get(rid)
            if probe is not None and hasattr(owner, 'forward_counts'):
                kind = 'audit' if probe['audit'] else 'probe'
                counts = owner.forward_counts[kind]
                counts['request_forward_rows'] += 1
                counts['input_tokens'] += int(batch.num_scheduled_tokens[pos])
                if kind not in counted:
                    counts['model_batches'] += 1
                    counted.add(kind)
            valid = (batch.num_computed_tokens_np[pos]
                     + batch.num_scheduled_tokens[pos] >= batch.prefill_len_np[pos])
            if not valid:
                continue
            probe = owner.probes.get(rid)
            if probe is not None:
                if probe['audit']:
                    if probe['logits'] is not None:
                        raise RuntimeError('Replay audit must generate one token')
                    probe['logits'] = logits[pos].detach().float().cpu()
                else:
                    f = logits[pos].float()
                    log_z = torch.logsumexp(f, -1)
                    p = torch.exp(f - log_z)
                    h = log_z - (p * f.masked_fill(~torch.isfinite(f), 0)).sum()
                    if not bool(torch.isfinite(h)):
                        raise RuntimeError('Nonfinite probe entropy')
                    probe['entropy'].append(float(h.clamp_min(0).item()))
                    probe['vocab_size'] = logits.shape[-1]
                continue
            if rid in owner.expected_logits:
                expected = owner.expected_logits.pop(rid).to(logits.device)
                actual = logits[pos].float()
                difference = actual - expected
                error = float(difference.abs().max().item())
                relative = float((difference.norm() / expected.norm().clamp_min(1e-12)).item())
                same_top = int(actual.argmax()) == int(expected.argmax())
                check = dict(request_id=rid, max_abs=error, relative_l2=relative,
                             argmax_equal=same_top,
                             passed=error <= .02 and relative <= .002 and same_top)
                owner.replay_checks.append(check)
                if not check['passed']:
                    raise RuntimeError(f'Historical R replay gate failed: {check}')
            state = owner.states.get(rid)
            if state and owner.suppress and state.should_mask():
                logits[pos, list(TRIGGERS)] = -float('inf')
                state.mask_count += 1
                state.mask_positions.append(state.count)
        owner.callback_host_seconds += time.perf_counter() - start
        # Runner has already cached raw R max probability before this callback.
        return self.original(logits, batch, **kwargs)


class Backend:
    def __init__(self, llm, suppress, branch_mode='replay', verify_cache=False):
        import torch
        self.torch, self.llm, self.suppress = torch, llm, suppress
        self.core = llm.llm_engine.engine_core.engine_core
        self.scheduler = self.core.scheduler
        self.runner = self.core.model_executor.driver_worker.worker.model_runner
        if branch_mode not in ('replay', 'kv_clone'):
            raise ValueError('Unknown branch mode')
        self.branch_mode, self.verify_cache = branch_mode, verify_cache
        self.clone_checks, self.clone_bytes = [], 0
        cfg = self.runner.vllm_config
        if (cfg.scheduler_config.async_scheduling or self.core.batch_queue is not None
                or cfg.parallel_config.tensor_parallel_size != 1
                or cfg.cache_config.enable_prefix_caching
                or cfg.speculative_config is not None):
            raise RuntimeError('C-probe v1 requires sync TP1, no prefix cache/speculation')
        if getattr(self.runner, 'hybrid_termination', None) is not None:
            raise RuntimeError('Disable S64/mix05 before C-probe')
        if not self.runner.steer_vector_state.supports_kv_replay:
            raise RuntimeError('Missing per-position R replay history')
        self.states, self.probes, self.expected_logits = {}, {}, {}
        self.replay_checks = []
        self.forward_counts = {kind: dict(request_forward_rows=0, model_batches=0,
                                         input_tokens=0) for kind in ('probe', 'audit')}
        self.callback_host_seconds = 0.
        self.original_sampler = self.runner.sampler
        self.original_add = self.runner.add_requests
        self.original_update = self.runner.update_requests
        self.original_preempt = self.scheduler._preempt_request
        self.runner.sampler = SamplerAdapter(self, self.original_sampler)
        self.runner.add_requests = self.add_requests
        if branch_mode == 'kv_clone':
            groups = self.runner.kv_cache_config.kv_cache_groups
            if (len(groups) != 1 or type(groups[0].kv_cache_spec).__name__ != 'FullAttentionSpec'
                    or groups[0].kv_cache_spec.block_size != self.runner.kernel_block_sizes[0]):
                raise RuntimeError('KV clone currently requires one unsplit full-attention group')
            self.block_size = groups[0].kv_cache_spec.block_size
            self.cache_views = self.cache_block_views()
            self.runner.update_requests = self.update_requests
        self.scheduler._preempt_request = self.reject_preemption

    @staticmethod
    def reject_preemption(*args, **kwargs):
        raise RuntimeError('C-probe v1 has not validated KV eviction; stop batch')

    def close(self):
        self.runner.sampler = self.original_sampler
        self.runner.add_requests = self.original_add
        self.runner.update_requests = self.original_update
        self.scheduler._preempt_request = self.original_preempt

    def add_requests(self, output):
        self.original_add(output)
        state = self.runner.steer_vector_state
        for request in output.scheduled_new_reqs:
            probe = self.probes.get(request.req_id)
            if probe is None or probe['snapshot'] is None:
                continue
            saved = probe['snapshot']
            idx = state._dynamic_indices[request.req_id]
            if len(request.prompt_token_ids) < len(saved['history']):
                raise RuntimeError('Probe prefix shorter than R history')
            state._history[idx].zero_()
            state._history[idx, :len(saved['history'])].copy_(saved['history'])
            state._prompt_lengths[request.req_id] = saved['prompt_length']
            state._history_lengths[request.req_id] = len(request.prompt_token_ids)
            # Reuse historical steering only. Probe descendants add no new R.
            state._in_think[idx] = False
            state._coefs[idx] = 0.

    def cache_block_views(self):
        """Use the pinned runner's block-major storage, including packed K/V."""
        seen, views = set(), []
        count = self.runner.kv_cache_config.num_blocks
        for tensor in self.runner.kv_caches:
            if not isinstance(tensor, self.torch.Tensor):
                raise RuntimeError('Unsupported recurrent/mixed cache')
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            raw = self.torch.empty(0, dtype=self.torch.uint8, device=tensor.device)
            raw.set_(storage)
            if raw.numel() % count:
                raise RuntimeError('Non-block-major cache storage')
            views.append(raw.view(count, -1))
        if not views:
            raise RuntimeError('No KV storage')
        return views

    def update_requests(self, output):
        # The original method zeros newly allocated blocks. Copy AFTER zeroing.
        self.original_update(output)
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace
        pairs = []
        for req in output.scheduled_new_reqs:
            probe = self.probes.get(req.req_id)
            if probe is None:
                continue
            parent = self.scheduler.requests[probe['parent']]
            n = probe['cached_tokens']
            if parent.num_computed_tokens != n or req.num_computed_tokens != n:
                raise RuntimeError('Parent or child KV prefix moved before copy')
            if list(req.prompt_token_ids[:n]) != list(parent.all_token_ids[:n]):
                raise RuntimeError('KV copy token identity differs')
            count = (n + self.block_size - 1) // self.block_size
            src = self.scheduler.kv_cache_manager.get_block_ids(parent.request_id)[0][:count]
            dst = req.block_ids[0][:count]
            if len(src) != count or len(dst) != count or set(src) & set(dst):
                raise RuntimeError('KV copy must own disjoint complete block allocations')
            pairs.extend(zip(src, dst))
        if not pairs:
            return
        if len({d for _, d in pairs}) != len(pairs):
            raise RuntimeError('Aliased child KV blocks')
        copy_kv_cache_blocks_inplace(self.runner.kv_caches,
                                    self.runner.kv_cache_config.num_blocks, pairs)
        src = self.torch.tensor([s for s, _ in pairs], device=self.cache_views[0].device)
        dst = self.torch.tensor([d for _, d in pairs], device=src.device)
        copied = sum(v.shape[1] * len(pairs) for v in self.cache_views)
        self.clone_bytes += copied
        verified = None
        if self.verify_cache:
            verified = all(self.torch.equal(v[src], v[dst]) for v in self.cache_views)
            if not verified:
                raise RuntimeError('Cloned KV differs byte-for-byte before forward')
        self.clone_checks.append(dict(block_pairs=len(pairs), bytes=copied,
                                      byte_equal=verified))

    def snapshot(self, rid, prefix_length):
        state = self.runner.steer_vector_state
        if rid not in state._dynamic_indices:
            return None
        idx = state._dynamic_indices[rid]
        if state._history_lengths[rid] != prefix_length:
            raise RuntimeError('R history is not aligned to accepted main tokens')
        return dict(prompt_length=state._prompt_lengths[rid],
                    history=state._history[idx, :prefix_length].clone())

    def primary_signature(self, held):
        state = self.runner.steer_vector_state
        signatures = {}
        bulk = getattr(self, 'bulk_signature', False)
        cached = {}
        if bulk:
            routed = [r.request_id for r in held if r.request_id in state._dynamic_indices]
            if routed:
                index = self.torch.tensor([state._dynamic_indices[r] for r in routed],
                                         device=state._history.device)
                fields = [f.index_select(0, index).cpu().numpy() for f in state._state_fields()]
                length = max(state._history_lengths[r] for r in routed)
                histories = state._history[:, :length].index_select(0, index).cpu().numpy()
                cached = {rid: ([f[i].tobytes() for f in fields],
                                histories[i, :state._history_lengths[rid]].tobytes())
                          for i, rid in enumerate(routed)}
        for req in held:
            rid = req.request_id
            h = hashlib.sha256()
            h.update(bytes(str(list(req.all_token_ids)), 'utf8'))
            h.update(str((req.num_computed_tokens, req.num_output_tokens)).encode())
            if getattr(self, 'verify_cache', False) and self.branch_mode == 'kv_clone':
                blocks = self.scheduler.kv_cache_manager.get_block_ids(rid)[0]
                for view in self.cache_views:
                    h.update(view[blocks].cpu().numpy().tobytes())
            if rid in state._dynamic_indices:
                idx = state._dynamic_indices[rid]
                if bulk:
                    fields, history = cached[rid]
                    for value in fields:
                        h.update(value)
                    h.update(history)
                else:
                    for field in state._state_fields():
                        h.update(field[idx].cpu().numpy().tobytes())
                    h.update(state._history[idx, :state._history_lengths[rid]]
                             .cpu().numpy().tobytes())
            signatures[rid] = h.hexdigest()
        return signatures

    def drain_probe_requests(self, requests, sampling, steering, suffix, audit=False):
        """Caller parks primary scheduler list before invoking this method."""
        limit = getattr(self, 'probe_batch_limit', None)
        if limit is not None and requests:
            free = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            selected, required = 0, 0
            for item in requests[:limit]:
                blocks = (len(item['prefix']) + len(suffix) + sampling.max_tokens
                          + self.block_size - 1) // self.block_size
                if required + blocks > free - 32:
                    break
                selected += 1
                required += blocks
            if not selected:
                raise RuntimeError('Insufficient independent probe KV capacity; stop without skipping probe')
            if selected < len(requests):
                return (self.drain_probe_requests(requests[:selected], sampling, steering, suffix, audit)
                        + self.drain_probe_requests(requests[selected:], sampling, steering, suffix, audit))
        prompts = [{'prompt_token_ids': item['prefix'] + suffix} for item in requests]
        ids = self.llm.enqueue(prompts, sampling_params=sampling, steering=steering,
                               use_tqdm=False)
        enable_cumulative(self.llm, ids)
        output_states = self.llm.llm_engine.output_processor.request_states
        external = {output_states[rid].external_req_id: rid for rid in ids}
        for rid, item in zip(ids, requests):
            self.probes[rid] = dict(snapshot=item['snapshot'], entropy=[], logits=None,
                                    audit=audit, parent=item['rid'])
            if getattr(self, 'branch_mode', 'replay') == 'kv_clone':
                parent = self.scheduler.requests[item['rid']]
                cached = parent.num_computed_tokens
                if cached != len(item['prefix']) - 1 or cached <= 0:
                    raise RuntimeError('Clone needs the live prefix with one pending token')
                child = self.scheduler.requests[rid]
                child.num_computed_tokens = cached
                self.probes[rid]['cached_tokens'] = cached
        results = {}
        while len(results) < len(ids):
            if time.monotonic() > self.deadline:
                raise TimeoutError('Probe wall limit')
            for result in self.llm.llm_engine.step():
                if result.request_id not in external:
                    raise RuntimeError('Parked main request produced output')
                rid = external[result.request_id]
                probe = self.probes[rid]
                completion = result.outputs[0]
                complete_box = False
                if not audit:
                    _, status, _ = boxed_certainty(
                        list(completion.token_ids), probe['entropy'],
                        self.tokenizer, probe['vocab_size'])
                    complete_box = status in ('complete', 'no_interior_tokens')
                if result.finished or complete_box:
                    if not audit and len(probe['entropy']) != len(completion.token_ids):
                        raise RuntimeError('Probe entropy/output alignment failed')
                    results[rid] = dict(probe, token_ids=list(completion.token_ids),
                                        finish_reason='complete_box' if complete_box
                                        else completion.finish_reason)
                    if not result.finished:
                        self.llm.llm_engine.abort_request([rid], internal=True)
        # Worker consumes finished IDs on the following step, including slot release.
        self.llm.llm_engine.step()
        for rid in ids:
            if rid in self.runner.req_states.req_id_to_index:
                # An empty core step may not execute a worker batch. Synchronous
                # completion means it is safe to release that finished slot now.
                self.runner._remove_request(rid)
            self.probes.pop(rid)
        return [results[rid] for rid in ids]
