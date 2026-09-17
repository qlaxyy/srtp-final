"""Instance-only eager engineering observer. Never modifies model outputs.

Install after RC14 Adapter, close before it. No compile/CUDA graphs, async,
preemption, speculative tokens or slot reuse. These are engineering limits,
not a production performance recommendation. Scoring happens after generation.
"""


class WSCShadow:
    def __init__(self, runner, *, enabled=False, max_calls=512):
        self.enabled = enabled
        if not enabled:
            return  # No imports, hooks, buffers, or runner inspection.
        import torch
        cfg = runner.vllm_config
        if (not cfg.model_config.enforce_eager or int(cfg.compilation_config.mode) != 0 or
                cfg.parallel_config.tensor_parallel_size != 1 or
                cfg.parallel_config.pipeline_parallel_size != 1 or
                cfg.speculative_config is not None or
                cfg.scheduler_config.async_scheduling or
                cfg.cache_config.enable_prefix_caching):
            raise ValueError('WSC capture requires synchronous eager TP1/PP1 without speculation/cache')
        if not 1 <= max_calls <= 512:
            raise ValueError('Engineering capture limit')
        self.runner, self.original = runner, runner.sampler
        self.model = runner.model.model
        if len(self.model.layers) != 28:
            raise ValueError('Fixed 28-layer Qwen contract')
        self.torch, self.max_calls = torch, max_calls
        self.frames, self.pending, self.handles = [], None, []
        self.handles.append(self.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
        self.handles.append(self.model.layers[26].register_forward_hook(self.after_layer))
        self.handles.append(self.model.layers[27].register_forward_pre_hook(self.before_last))
        runner.sampler = self

    def __getattr__(self, name):
        original = self.__dict__.get('original')
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    def before_model(self, module, args, kwargs):
        if self.pending is not None:
            raise RuntimeError('Forward without consuming previous observation')
        ids = kwargs.get('input_ids', args[0] if args else None)
        pos = kwargs.get('positions', args[1] if len(args) > 1 else None)
        if ids is None or pos is None or ids.ndim != 1 or pos.ndim != 1:
            raise ValueError('Expected actual flat input IDs and positions')
        self.pending = dict(ids=ids.clone(), positions=pos.clone())

    def after_layer(self, module, args, output):
        if self.pending is None or 'hidden' in self.pending:
            raise RuntimeError('Missing or duplicate HF27 boundary')
        hidden, residual = output
        if residual is None or hidden.shape != residual.shape:
            raise ValueError('Expected Qwen hidden/residual pair')
        # Add in native dtype before FP32 export, matching the raw stream.
        self.pending['hidden'] = (hidden + residual).clone()

    def before_last(self, module, args):
        if self.pending is None or 'hidden' not in self.pending:
            raise RuntimeError('Layer27 input arrived without layer26 output')
        _, hidden, residual = args
        reference = hidden + residual
        self.torch._assert_async((reference == self.pending['hidden']).all(),
                                'HF27 residual boundary changed before final block')
        self.pending['checked'] = True

    def __call__(self, logits, batch, **kwargs):
        t, frame = self.torch, self.pending
        if len(self.frames) >= self.max_calls or frame is None or not frame.get('checked'):
            raise RuntimeError('Missing capture or capture limit')
        if batch.num_draft_tokens or batch.num_reqs > 8 or logits.shape[0] != batch.num_reqs:
            raise ValueError('At most eight one-token requests')
        take = batch.logits_indices.long()
        if take.numel() != batch.num_reqs:
            raise ValueError('Unexpected logits index mapping')
        idx = batch.idx_mapping[:batch.num_reqs].long().clone()
        valid = t.as_tensor(batch.num_computed_tokens_np + batch.num_scheduled_tokens
                            >= batch.prefill_len_np, device=logits.device).clone()
        # Copy before sampler/controller updates any mutable buffers.
        pos, ids, hidden = (frame[k][take].clone() for k in ('positions','ids','hidden'))
        t._assert_async((~valid | (pos == batch.seq_lens[:batch.num_reqs]-1)).all(),
                        'Input position does not match request length')
        result = self.original(logits, batch, **kwargs)
        if result.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise ValueError('Unexpected sampler output')
        self.frames.append((idx,valid,pos,ids,hidden,result.sampled_token_ids[:,0].clone()))
        self.pending = None
        return result

    def export(self):
        rows = {}
        for tensors in self.frames:
            idx, valid, pos, ids, hidden, selected = [
                x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
                for x in tensors]
            for i, slot in enumerate(idx):
                if valid[i]:
                    rows.setdefault(int(slot), []).append(dict(position=int(pos[i]),
                        input_id=int(ids[i]), hidden=hidden[i], selected=int(selected[i])))
        return rows

    def close(self):
        if not self.enabled:
            return
        if self.runner.sampler is not self:
            raise RuntimeError('Close WSC before RC14 adapter')
        self.runner.sampler = self.original
        for handle in self.handles:
            handle.remove()
        self.pending = None
