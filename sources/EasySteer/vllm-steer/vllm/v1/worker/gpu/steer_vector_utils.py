# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner."""

import numpy as np
import torch

from vllm.steer_vectors import trace
from vllm.steer_vectors.rebalance import (
    ReBalanceParams,
    compute_rebalance_coefficient,
)
from vllm.steer_vectors.request import SteerVectorRequest
from vllm.v1.worker.gpu.input_batch import InputBatch


class SteerVectorState:
    """Per-request steer vector bookkeeping for the V2 model runner.

    Each live request resolves to a config slot at admission time
    (payload loading + layer distribution happen there, never in the
    forward pass).
    """

    def __init__(
        self,
        max_num_reqs: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._requests: dict[str, SteerVectorRequest] = {}
        self._slots: dict[str, int] = {}
        self._dynamic_params: dict[str, ReBalanceParams] = {}
        self._dynamic_indices: dict[str, int] = {}
        self._boundary_tensors: dict[ReBalanceParams, torch.Tensor] = {}
        self._coefs: torch.Tensor | None = None
        self._step_prob_sum: torch.Tensor | None = None
        self._step_tok_count: torch.Tensor | None = None
        self._prev_step_mean: torch.Tensor | None = None
        self._in_think: torch.Tensor | None = None
        if max_num_reqs is not None and device is not None:
            self._allocate_dynamic_state(max_num_reqs, device)

    def _allocate_dynamic_state(
        self, max_num_reqs: int, device: torch.device
    ) -> None:
        self._coefs = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self._step_prob_sum = torch.zeros_like(self._coefs)
        self._step_tok_count = torch.zeros(
            max_num_reqs, dtype=torch.long, device=device
        )
        self._prev_step_mean = torch.full_like(self._coefs, torch.nan)
        self._in_think = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )

    def add_request(
        self,
        req_id: str,
        steer_vector_request: SteerVectorRequest | None,
        manager,
        *,
        req_index: int | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        if steer_vector_request is None:
            return
        if manager is None:
            # Admission should have rejected this; routing the request to
            # the server config (or to nothing) would silently steer with
            # the wrong vector.
            raise RuntimeError(
                f"request {req_id} carries a steering config but this "
                "worker has no steer vector manager (engine launched "
                "without enable_steer_vector=True)"
            )
        self._requests[req_id] = steer_vector_request
        self._slots[req_id] = manager.acquire_config(req_id, steer_vector_request)
        if steer_vector_request.algorithm != "rebalance":
            return
        if req_index is None or self._coefs is None:
            raise RuntimeError("rebalance requires initialized request-index state")
        params = ReBalanceParams.from_request(steer_vector_request)
        self._dynamic_params[req_id] = params
        self._dynamic_indices[req_id] = req_index
        self._coefs[req_index] = params.initial_coef
        self._step_prob_sum[req_index] = 0.0
        self._step_tok_count[req_index] = 0
        self._prev_step_mean[req_index] = torch.nan
        prompt_token_ids = prompt_token_ids or []
        self._in_think[req_index] = params.think_start_token_id in prompt_token_ids

    def remove_request(self, req_id: str, manager) -> None:
        if self._requests.pop(req_id, None) is None:
            return
        self._slots.pop(req_id, None)
        self._dynamic_params.pop(req_id, None)
        req_index = self._dynamic_indices.pop(req_id, None)
        if req_index is not None and self._coefs is not None:
            self._coefs[req_index] = 0.0
            self._step_prob_sum[req_index] = 0.0
            self._step_tok_count[req_index] = 0
            self._prev_step_mean[req_index] = torch.nan
            self._in_think[req_index] = False
        if manager is not None:
            manager.release_config(req_id)

    def slot_of(self, req_id: str) -> int:
        return self._slots.get(req_id, -1)

    def has_routed(self) -> bool:
        return bool(self._slots)

    def has_dynamic(self) -> bool:
        return bool(self._dynamic_params)

    def _group_batch_positions(
        self, req_ids: list[str]
    ) -> dict[ReBalanceParams, list[int]]:
        groups: dict[ReBalanceParams, list[int]] = {}
        for position, req_id in enumerate(req_ids):
            params = self._dynamic_params.get(req_id)
            if params is not None:
                groups.setdefault(params, []).append(position)
        return groups

    def batch_scales(self, input_batch: InputBatch) -> torch.Tensor:
        """Return one online coefficient per request in scheduler order."""
        device = input_batch.idx_mapping.device
        scales = torch.ones(input_batch.num_reqs, dtype=torch.float32, device=device)
        if not self.has_dynamic():
            return scales
        assert self._coefs is not None and self._in_think is not None
        for positions in self._group_batch_positions(input_batch.req_ids).values():
            pos = torch.tensor(positions, dtype=torch.long, device=device)
            state_idx = input_batch.idx_mapping.index_select(0, pos).long()
            active_scales = self._coefs.index_select(0, state_idx)
            active_scales *= self._in_think.index_select(0, state_idx)
            scales.index_copy_(0, pos, active_scales)
        return scales

    def token_scales(self, input_batch: InputBatch) -> torch.Tensor:
        """Expand request coefficients to the flattened input-token rows."""
        request_scales = self.batch_scales(input_batch)
        repeats = torch.diff(
            input_batch.query_start_loc[: input_batch.num_reqs + 1]
        ).long()
        return torch.repeat_interleave(request_scales, repeats)

    def observe_sample(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,
        max_probabilities: torch.Tensor,
    ) -> None:
        """Update each request after sampling without synchronizing to the CPU."""
        if not self.has_dynamic():
            return
        if input_batch.num_draft_tokens != 0:
            raise RuntimeError("rebalance does not support speculative decoding")
        if sampled_token_ids.shape != (input_batch.num_reqs, 1):
            raise RuntimeError(
                "rebalance expects exactly one sampled token per request"
            )
        if max_probabilities.shape != (input_batch.num_reqs,):
            raise RuntimeError("rebalance confidence rows do not match the batch")

        assert self._coefs is not None
        assert self._step_prob_sum is not None
        assert self._step_tok_count is not None
        assert self._prev_step_mean is not None
        assert self._in_think is not None
        device = sampled_token_ids.device
        sampled = sampled_token_ids[:, 0]

        for params, positions in self._group_batch_positions(
            input_batch.req_ids
        ).items():
            pos = torch.tensor(positions, dtype=torch.long, device=device)
            state_idx = input_batch.idx_mapping.index_select(0, pos).long()
            tokens = sampled.index_select(0, pos)
            probabilities = max_probabilities.index_select(0, pos)
            boundaries = self._boundary_tensors.get(params)
            if boundaries is None:
                boundaries = torch.tensor(
                    params.boundary_token_ids,
                    dtype=tokens.dtype,
                    device=device,
                )
                self._boundary_tensors[params] = boundaries
            is_boundary = torch.isin(tokens, boundaries)

            in_think = self._in_think.index_select(0, state_idx)
            in_think |= tokens == params.think_start_token_id
            in_think &= tokens != params.think_end_token_id
            self._in_think.index_copy_(0, state_idx, in_think)

            not_boundary = ~is_boundary
            sums = self._step_prob_sum.index_select(0, state_idx)
            counts = self._step_tok_count.index_select(0, state_idx)
            sums += probabilities * not_boundary
            counts += not_boundary
            self._step_prob_sum.index_copy_(0, state_idx, sums)
            self._step_tok_count.index_copy_(0, state_idx, counts)

            ready = is_boundary & (counts > 0)
            step_means = sums / counts.clamp_min(1).float()
            previous = self._prev_step_mean.index_select(0, state_idx)
            variance = torch.where(
                torch.isfinite(previous),
                (step_means - previous).square() / 4.0,
                torch.zeros_like(step_means),
            )
            updated = compute_rebalance_coefficient(
                step_means, variance, params
            )
            current = self._coefs.index_select(0, state_idx)
            self._coefs.index_copy_(0, state_idx, torch.where(ready, updated, current))
            self._step_prob_sum.index_copy_(
                0, state_idx, torch.where(ready, torch.zeros_like(sums), sums)
            )
            self._step_tok_count.index_copy_(
                0, state_idx, torch.where(ready, torch.zeros_like(counts), counts)
            )
            self._prev_step_mean.index_copy_(
                0,
                state_idx,
                torch.where(ready, step_means.detach(), previous),
            )


def build_batch_geometry(input_batch: InputBatch) -> "BatchGeometry":
    """Build the per-step BatchGeometry from the runner's InputBatch.

    The single producer of batch geometry: steering triggers, capture
    row selection/labels, and the full-graph buffer filler all consume
    this object (directly or via `geometry_samples_info`).
    """
    from vllm.forward_context import BatchGeometry

    num_reqs = input_batch.num_reqs
    num_computed = input_batch.num_computed_tokens_np[:num_reqs]
    prefill_len = input_batch.prefill_len_np[:num_reqs]
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    # While prefilling, nothing has been generated for this request yet.
    # During decode, the scheduler has computed prefill_len + (k - 1) tokens
    # when the k-th output token is being generated, matching the V1
    # semantics of len(output_token_ids) at execute time.
    num_output = np.where(is_prefilling, 0, num_computed - prefill_len + 1).astype(
        np.int32
    )
    return BatchGeometry(
        query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
        num_computed=torch.from_numpy(np.ascontiguousarray(num_computed)),
        num_prompt=torch.from_numpy(np.ascontiguousarray(prefill_len)),
        num_output=torch.from_numpy(num_output),
        req_ids=list(input_batch.req_ids[:num_reqs]),
        token_ids=input_batch.input_ids[: input_batch.num_tokens],
        query_start_loc_cpu=input_batch.query_start_loc_np[: num_reqs + 1],
    )


def _batch_token_slots(
    input_batch: InputBatch, state: SteerVectorState, default_slot: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-request and per-token config-slot routing for this step."""
    num_reqs = input_batch.num_reqs
    slots_np = np.fromiter(
        (
            slot if (slot := state.slot_of(req_id)) >= 0 else default_slot
            for req_id in input_batch.req_ids
        ),
        dtype=np.int32,
        count=num_reqs,
    )
    token_slots_np = np.repeat(slots_np, input_batch.num_scheduled_tokens[:num_reqs])
    return slots_np, token_slots_np


def _match_positions_np(
    abs_pos: np.ndarray, positions, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Mask of prompt tokens at the given positions (numpy mirror of
    clause._match_positions): negative entries index from each sample's
    prompt length; positive entries past the prompt end clamp to the
    last prompt token; decode tokens never match."""
    mask = np.zeros(abs_pos.shape[0], dtype=bool)
    for p in positions:
        if p < 0:
            mask |= abs_pos == neg_base + p
        else:
            mask |= abs_pos == np.minimum(neg_base - 1, p)
    return mask & ~is_dec


def _match_prompt_window_np(
    abs_pos: np.ndarray, window, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Mask of prompt tokens inside the half-open window (numpy mirror
    of clause._match_prompt_window): negative bounds resolve from each
    sample's prompt length; stop=None means the prompt end."""
    start, stop = window
    lo = neg_base + start if start < 0 else start
    hi = neg_base if stop is None else (neg_base + stop if stop < 0 else stop)
    return ~is_dec & (abs_pos >= lo) & (abs_pos < hi)


def _match_generation_steps_np(
    gen_idx: np.ndarray, is_dec: np.ndarray, steps, window
) -> np.ndarray:
    """Mask of generation tokens at the given 0-based decode steps
    and/or inside the half-open decode-step window (numpy mirror of
    clause._match_generation_steps)."""
    mask = np.zeros(gen_idx.shape[0], dtype=bool)
    if steps is not None:
        mask |= np.isin(gen_idx, np.asarray(list(steps), dtype=np.int64))
    if window is not None:
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask |= in_window
    return mask & is_dec


def _clause_mask_np(
    clause: dict,
    is_dec: np.ndarray,
    abs_pos: np.ndarray,
    neg_base: np.ndarray,
    gen_idx: np.ndarray,
    token_ids,
) -> np.ndarray:
    """Evaluate one where-clause over a slot's tokens (numpy mirror of
    clause.collect_positions_apply_spec — same union/veto semantics).

    `token_ids` is a thunk: only token-id filters pay for the host copy.
    """
    n = is_dec.shape[0]

    def _selector_mask(
        prompt_tokens,
        prompt_positions,
        prompt_window,
        generation_tokens,
        generation_positions,
        generation_window,
    ) -> np.ndarray:
        matched = np.zeros(n, dtype=bool)
        if prompt_tokens is not None:
            matched |= (
                np.isin(token_ids(), np.asarray(list(prompt_tokens))) & ~is_dec
            )
        if generation_tokens is not None:
            matched |= (
                np.isin(token_ids(), np.asarray(list(generation_tokens))) & is_dec
            )
        if prompt_positions is not None:
            matched |= _match_positions_np(
                abs_pos, prompt_positions, neg_base, is_dec
            )
        if prompt_window is not None:
            matched |= _match_prompt_window_np(abs_pos, prompt_window, neg_base, is_dec)
        if generation_positions is not None or generation_window is not None:
            matched |= _match_generation_steps_np(
                gen_idx, is_dec, generation_positions, generation_window
            )
        return matched

    mask = np.zeros(n, dtype=bool)
    if clause.get("prompt") == "all":
        mask |= ~is_dec
    if clause.get("generation") == "all":
        mask |= is_dec

    from vllm.steer_vectors.algorithms.clause import _EXCLUDE_KEYS, _INCLUDE_KEYS

    includes = tuple(clause.get(key) for key in _INCLUDE_KEYS)
    if any(value is not None for value in includes):
        mask |= _selector_mask(*includes)

    excludes = tuple(clause.get(key) for key in _EXCLUDE_KEYS)
    if any(value is not None for value in excludes):
        mask &= ~_selector_mask(*excludes)
    return mask


def resolve_slot_positions(
    slot_clauses: dict[int, list[dict | None]],
    active_slots: list[int],
    token_slots_np: np.ndarray,
    device: torch.device,
    geo,
) -> dict[tuple, torch.Tensor | None]:
    """Resolve every active clause's steered positions, once per step.

    Where-clauses are layer-invariant, so this single resolution serves
    every decoder/MoE-gate hook (and the Tier-1 mask filler). Keys are
    (slot, clause_cache_key); a None value means the clause matched no
    token this step.

    Resolution runs host-side in one numpy pass: clauses match phases,
    positions and windows — all host-known geometry — so each slot's
    clauses are evaluated only over that slot's own token rows and the
    matched positions ship to the device in a single copy. Per-step cost
    scales with the batch's tokens, not with the number of distinct live
    configurations. Only token-id filters read the input ids (one cached
    device-to-host copy per step).
    """
    from vllm.steer_vectors.algorithms.clause import (
        clause_cache_key,
        selects_all_tokens,
    )

    resolved: dict[tuple, torch.Tensor | None] = {}
    if not active_slots:
        return resolved

    qsl = geo.query_start_loc_cpu
    assert qsl is not None, "BatchGeometry is missing its host query_start_loc"
    num_computed = geo.num_computed.numpy()
    num_prompt = geo.num_prompt.numpy()
    num_output = geo.num_output.numpy()
    lens = (qsl[1:] - qsl[:-1]).astype(np.int64)
    starts_all = qsl[:-1].astype(np.int64)
    is_decode_req = num_output > 0

    # Group batch requests by routing slot (a request's slot is its
    # first token's slot; all its tokens share it).
    active = set(active_slots)
    slot_reqs: dict[int, list[int]] = {}
    for r, s in enumerate(token_slots_np[starts_all].tolist()):
        if s in active:
            slot_reqs.setdefault(s, []).append(r)

    keys: list[tuple[int, tuple]] = []
    chunks: list[np.ndarray] = []

    for slot in active_slots:
        reqs = slot_reqs.get(slot)
        clauses = slot_clauses.get(slot, [])
        if not reqs:
            for clause in clauses:
                key = clause_cache_key(clause)
                if key is not None:
                    resolved.setdefault((slot, key), None)
            continue
        reqs_np = np.asarray(reqs, dtype=np.int64)
        seg_lens = lens[reqs_np]
        n = int(seg_lens.sum())
        samp = np.repeat(reqs_np, seg_lens)
        within = np.arange(n, dtype=np.int64) - np.repeat(
            np.cumsum(seg_lens) - seg_lens, seg_lens
        )
        tok_idx = np.repeat(starts_all[reqs_np], seg_lens) + within
        abs_pos = within + num_computed[samp]
        is_dec = is_decode_req[samp]

        for clause in clauses:
            key = clause_cache_key(clause)
            if key is None or (slot, key) in resolved:
                continue
            if selects_all_tokens(clause):
                pos_np = tok_idx
            else:
                mask = _clause_mask_np(
                    clause,
                    is_dec,
                    abs_pos,
                    num_prompt[samp],
                    num_output[samp] - 1,
                    lambda: geo.token_ids_cpu()[tok_idx],
                )
                pos_np = tok_idx[mask]
            if pos_np.shape[0] == 0:
                resolved[(slot, key)] = None
            else:
                resolved[(slot, key)] = pos_np  # placeholder, replaced below
                keys.append((slot, key))
                chunks.append(pos_np)

    if chunks:
        flat = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        flat_t = torch.from_numpy(flat).to(device, non_blocking=True)
        offset = 0
        for key, chunk in zip(keys, chunks):
            size = chunk.shape[0]
            resolved[key] = flat_t[offset : offset + size]
            offset += size
    return resolved


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
    state: SteerVectorState | None = None,
    default_slot: int = -1,
    manager=None,
) -> dict:
    """Build the ForwardContext fields consumed by steering and capture.

    - batch_geometry: the per-step BatchGeometry (see build_batch_geometry)
    - steer_token_slots / steer_active_slots: per-request config routing
      (only when routed configs are live)
    - steer_slot_positions: per-clause steered positions, resolved once
      here and consumed by every layer hook (see resolve_slot_positions)

    `default_slot` is the server-level config's slot (-1 when absent);
    requests without their own steering config are routed to it.
    """
    num_reqs = input_batch.num_reqs
    geo = build_batch_geometry(input_batch)
    kwargs = {"batch_geometry": geo}

    if state is not None and (state.has_routed() or default_slot >= 0):
        slots_np, token_slots_np = _batch_token_slots(input_batch, state, default_slot)
        token_slots = torch.from_numpy(token_slots_np).to(
            input_batch.input_ids.device, non_blocking=True
        )
        active_slots = sorted({int(s) for s in slots_np if s >= 0})
        kwargs["steer_token_slots"] = token_slots
        if state.has_dynamic():
            kwargs["steer_token_scales"] = state.token_scales(input_batch)
        kwargs["steer_active_slots"] = active_slots
        if manager is None:
            raise RuntimeError(
                "steering slots are routed but no worker manager was passed "
                "to make_steer_vector_forward_kwargs"
            )
        kwargs["steer_slot_positions"] = resolve_slot_positions(
            manager.slot_clauses(),
            active_slots,
            token_slots_np,
            token_slots.device,
            geo,
        )

        if trace.enabled():
            trace.begin_step(
                req_ids=input_batch.req_ids,
                slots=slots_np.tolist(),
                query_start_loc=input_batch.query_start_loc_np[: num_reqs + 1].tolist(),
                token_ids=geo.token_ids.cpu().tolist(),
                num_computed=geo.num_computed.tolist(),
                num_output=geo.num_output.tolist(),
            )
    return kwargs


def fill_graph_steer_buffers(
    input_batch: InputBatch,
    state: SteerVectorState | None,
    manager,
) -> None:
    """Fill Tier-1 persistent buffers for this step (full-graph mode).

    Writes each token's vector-table row into the shared row buffer and
    sets the per-layer trigger masks to 1 at steered positions. The
    captured kernel `hidden += mask * vectors[row_tok]` then applies the
    right configs without any per-step graph work; row 0 / mask 0 keep
    unsteered and padding tokens untouched. Positions come from the same
    resolver the layer hooks use (resolve_slot_positions).
    """
    from vllm.steer_vectors.algorithms.clause import clause_cache_key

    manager.zero_graph_masks()
    row_buf = manager.token_rows_buf
    row_buf.zero_()
    entries = manager.graph_batch_entries()
    if not entries or state is None:
        return

    slots_np, token_slots_np = _batch_token_slots(
        input_batch, state, manager.server_slot
    )
    token_scales = state.token_scales(input_batch) if state.has_dynamic() else None
    rows_np = np.fromiter(
        (entries[s][0] if s in entries else 0 for s in slots_np),
        dtype=np.int64,
        count=slots_np.shape[0],
    )
    num_scheduled = input_batch.num_scheduled_tokens[: slots_np.shape[0]]
    token_rows_np = np.repeat(rows_np, num_scheduled)
    n = token_rows_np.shape[0]
    if n == 0:
        return
    device = row_buf.device
    row_buf[:n].copy_(torch.from_numpy(token_rows_np).to(device, non_blocking=True))

    # Batch geometry for the trigger collector: the same object the
    # forward context carries, from the single producer.
    geo = build_batch_geometry(input_batch)
    batch_slots = set(slots_np.tolist())
    active_slots = sorted(s for s in entries if s in batch_slots)
    resolved = resolve_slot_positions(
        manager.slot_clauses(), active_slots, token_slots_np, device, geo
    )
    from vllm.steer_vectors.algorithms import get_algorithm
    from vllm.steer_vectors.graph_kernels import graph_family_mask_attr

    # One scatter per (module, mask attr), not per slot: slots sharing a
    # layer contribute to the same mask write, so the launch count scales
    # with steered layers, not with live configurations.
    mask_writes: dict[tuple[int, str], tuple] = {}
    for slot in active_slots:
        _, request, controllers = entries[slot]
        positions = resolved[(slot, clause_cache_key(request.apply_spec))]
        if positions is None:
            continue
        # Families whose delta a zero table row cannot neutralize (e.g.
        # replace) carry their own mask; see GRAPH_FAMILY_MASKS.
        mask_attr = graph_family_mask_attr(
            get_algorithm(request.algorithm).graph_family
        )
        for module in controllers:
            mask_writes.setdefault(
                (id(module), mask_attr), (module, mask_attr, [])
            )[2].append(
                (
                    positions,
                    None
                    if request.algorithm != "rebalance"
                    else token_scales.index_select(0, positions),
                )
            )
    for module, mask_attr, position_entries in mask_writes.values():
        position_list = [entry[0] for entry in position_entries]
        positions = (
            position_list[0]
            if len(position_list) == 1
            else torch.cat(position_list)
        )
        mask = getattr(module, mask_attr)
        if any(values is not None for _, values in position_entries):
            value_list = [
                torch.ones(len(pos), dtype=mask.dtype, device=mask.device)
                if values is None
                else values.to(mask.dtype)
                for pos, values in position_entries
            ]
            values = value_list[0] if len(value_list) == 1 else torch.cat(value_list)
            mask.index_copy_(0, positions, values)
        else:
            mask[positions] = 1.0
