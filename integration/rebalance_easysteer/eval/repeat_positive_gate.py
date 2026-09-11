"""Experimental complete-step repeat gate; no model imports or fitted thresholds.

The runtime adapter wraps the frozen V2 state in the single-process evaluator.
It changes historical *applied* scales only, never the controller's coefficients.
Host token inspection synchronizes the device once per sampled batch when enabled.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def byte_pieces(tokenizer_path):
    data = json.loads(Path(tokenizer_path).read_text(encoding='utf-8'))
    if data['decoder']['type'] != 'ByteLevel':
        raise ValueError('The experimental repeat gate requires ByteLevel tokens')
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            chars.append(256 + extra)
            values.append(value)
            extra += 1
    inverse = {chr(c): b for c, b in zip(chars, values)}
    pieces = {i: bytes(inverse[c] for c in text)
              for text, i in data['model']['vocab'].items()}
    for item in data['added_tokens']:
        pieces[item['id']] = item['content'].encode('utf-8')
    return pieces


class CompleteStepRepeat:
    """A→B→A→B evidence, closed by the original boundary token IDs.

    Entire token bytes, including boundary-token symbols, are retained. Only
    outer ASCII whitespace is stripped. A runtime step is not a semantic label.
    """
    def __init__(self, boundary_ids, think_start, think_end, in_think=True):
        self.boundaries = frozenset(boundary_ids)
        self.think_start, self.think_end = think_start, think_end
        self.in_think, self.ended = in_think, False
        self.observed_tokens = 0
        self.content_count = 0
        self.pending = bytearray()
        self.pending_start = 0
        self.units, self.spans, self.symbols = [], [], []
        self.intern = {}
        self.active = None

    def observe(self, token_id, piece):
        self.observed_tokens += 1
        is_boundary = token_id in self.boundaries
        eligible = is_boundary and self.content_count > 0
        if is_boundary:
            self.content_count = 0
        else:
            self.content_count += 1
        if self.ended:
            return None
        if token_id == self.think_end:
            self.ended, self.in_think = True, False
            self.pending.clear()
            self.active = None
            return None
        if token_id == self.think_start:
            self.in_think = True
            self.pending.clear()
            self.pending_start = self.observed_tokens
            return None
        if not self.in_think:
            self.pending_start = self.observed_tokens
            return None
        self.pending.extend(piece)
        if not is_boundary:
            return None
        text = bytes(self.pending).strip()
        span = [self.pending_start, self.observed_tokens]
        self.pending.clear()
        self.pending_start = self.observed_tokens
        if not text:
            return None
        self.units.append(text)
        self.spans.append(span)
        self.symbols.append(self.intern.setdefault(text, len(self.intern)))
        n = len(self.symbols)
        if self.active is not None:
            period = self.active['period']
            if self.symbols[-1] != self.symbols[-1-period]:
                self.active = None
        if self.active is None:
            for period in range(1, n//2 + 1):
                if (self.symbols[-1] == self.symbols[-1-period]
                        and self.symbols[n-2*period:n-period]
                        == self.symbols[n-period:n]):
                    self.active = dict(start_unit=n-2*period, period=period,
                        detected_token=self.observed_tokens)
                    break
        if self.active is None:
            return None
        return dict(**self.active, observed_end_unit=n,
            completed_copies=(n-self.active['start_unit'])//self.active['period'],
            boundary_token_1based=self.observed_tokens, eligible_boundary=eligible)

    def snapshot(self):
        return deepcopy(self)


def applied_coefficient(original, repeat, mode):
    """Scalar reference for an applied-scale change, not a state update."""
    if mode not in ('off', 'shadow', 'cancel_positive'):
        raise ValueError('Unknown repeat-gate mode')
    if not math.isfinite(original):
        raise ValueError('Nonfinite original coefficient')
    return min(original, 0.0) if repeat and mode == 'cancel_positive' else original


class RepeatPositiveAdapter:
    """Opt-in bridge to the verified private V2 evaluator/state contract."""
    METHODS = ('add_request', 'remove_request', 'suspend_request',
               'discard_suspended', 'observe_sample')

    def __init__(self, state, pieces, mode):
        if mode not in ('shadow', 'cancel_positive'):
            raise ValueError('Enabled adapter mode must be shadow or cancel_positive')
        if not state.supports_kv_replay or state.has_dynamic():
            raise RuntimeError('Install before admission, with V2 steering history')
        if getattr(state, '_repeat_positive_adapter', None) is not None:
            raise RuntimeError('Repeat gate is already installed')
        self.state, self.pieces, self.mode = state, pieces, mode
        self.byte_boundaries = frozenset(i for i, piece in pieces.items()
                                        if b'\n\n' in piece)
        self.live, self.suspended, self.chunks = {}, {}, []
        self.sample_batches_copied = 0
        self.original = {name: getattr(state, name) for name in self.METHODS}
        for name in self.METHODS:
            setattr(state, name, getattr(self, name))
        state._repeat_positive_adapter = self

    def add_request(self, req_id, request, manager, **kwargs):
        if (request is not None and request.algorithm == 'rebalance'
                and getattr(request, 'rebalance_paper_parameters', None) is not None):
            raise ValueError('Repeat gate is only for the current code controller')
        if (request is not None and request.algorithm == 'rebalance'
                and frozenset(request.rebalance_boundary_token_ids or ())
                != self.byte_boundaries):
            raise RuntimeError('Repeat tokenizer does not match controller boundary IDs')
        self.original['add_request'](req_id, request, manager, **kwargs)
        params = self.state._dynamic_params.get(req_id)
        if params is None:
            return
        saved = self.suspended.pop(req_id, None)
        generated = kwargs.get('num_generated_tokens', 0)
        if saved is not None:
            if saved.observed_tokens != generated:
                raise RuntimeError('Repeat detector and restored KV history disagree')
            self.live[req_id] = saved
        elif generated:
            raise RuntimeError('Generated prefix has no saved repeat-detector state')
        else:
            self.live[req_id] = CompleteStepRepeat(params.boundary_token_ids,
                params.think_start_token_id, params.think_end_token_id,
                params.think_start_token_id in (kwargs.get('prompt_token_ids') or []))

    def remove_request(self, req_id, manager):
        self.original['remove_request'](req_id, manager)
        self.live.pop(req_id, None)

    def suspend_request(self, req_id):
        self.original['suspend_request'](req_id)
        if req_id in self.live:
            self.suspended[req_id] = self.live[req_id].snapshot()

    def discard_suspended(self, req_id):
        self.original['discard_suspended'](req_id)
        self.suspended.pop(req_id, None)

    def observe_sample(self, batch, sampled_token_ids, max_probabilities):
        # Original computation and history update must happen exactly once.
        self.original['observe_sample'](batch, sampled_token_ids, max_probabilities)
        positions = [p for p, req in enumerate(batch.req_ids) if req in self.live
            and (not hasattr(batch, 'prefill_len_np')
                or batch.num_computed_tokens_np[p] + batch.num_scheduled_tokens[p]
                >= batch.prefill_len_np[p])]
        if not positions:
            return
        tokens = sampled_token_ids[:, 0].detach().cpu().tolist()
        self.sample_batches_copied += 1
        indices, columns, events = [], [], []
        for position in positions:
            req_id = batch.req_ids[position]
            detector = self.live[req_id]
            token = tokens[position]
            event = detector.observe(token, self.pieces[token])
            expected = self.state._prompt_lengths[req_id] + detector.observed_tokens
            if self.state._history_lengths[req_id] != expected:
                raise RuntimeError('Repeat gate would address the wrong input position')
            if event is not None and event['eligible_boundary']:
                indices.append(self.state._dynamic_indices[req_id])
                columns.append(expected-1)
                events.append(dict(request_id=req_id, **event))
        if not events:
            return
        raw = self.state._history[indices, columns].detach().clone()
        if self.mode == 'cancel_positive':
            self.state._history[indices, columns] = raw.clamp(max=0.0)
        # Copy scalars to the host only at export; coefficients stay on-device here.
        self.chunks.append((events, raw))

    def reset_metrics(self):
        if self.live or self.suspended:
            raise RuntimeError('Cannot reset repeat metrics with unfinished requests')
        self.chunks.clear()
        self.sample_batches_copied = 0

    def export(self):
        events = []
        for metadata, values in self.chunks:
            for item, original in zip(metadata, values.detach().cpu().tolist()):
                applied = applied_coefficient(original, True, self.mode)
                events.append(dict(**item, original_scale=original, applied_scale=applied))
        return dict(mode=self.mode, sample_batches_copied=self.sample_batches_copied,
            repeat_boundaries=len(events),
            positive_opportunities=sum(e['original_scale'] > 1e-6 for e in events),
            changed_scales=sum(e['original_scale'] != e['applied_scale'] for e in events),
            events=events, note='CPU token inspection adds synchronization; GPU timing/behavior not inferred from CPU tests.')


def install_repeat_gate(state, tokenizer_path, mode='off'):
    if mode == 'off':
        return None
    pieces = byte_pieces(tokenizer_path)
    adapter = RepeatPositiveAdapter(state, pieces, mode)
    adapter.tokenizer_sha256 = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    adapter.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return adapter
