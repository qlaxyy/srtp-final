"""Opt-in harmonic aggregation, using the native accepted-token accumulator.

The native sum temporarily contains reciprocal probabilities. At a completed
step, replace its provisional mean/coefficient BEFORE writing control history.
The stored previous mean is always harmonic between calls. No shared runtime
changes, no extra forward, and no synchronous replay support.
"""
from label_alignment_adapter import AlignmentAdapter


class HarmonicAdapter(AlignmentAdapter):
    def __init__(self,llm,tokenizer,*,enabled=False,**kwargs):
        super().__init__(llm,tokenizer,enabled=enabled,**kwargs)
        if not enabled:return
        if not self.large_suppression or self.lexical_control:
            self.close();raise ValueError('Harmonic requires unchanged L27 suppression')
        from vllm.steer_vectors.rebalance import compute_rebalance_coefficient
        self.compute_coefficient=compute_rebalance_coefficient
        t=self.torch
        self.hready=t.zeros_like(self.lex_ready)
        self.hprevious=t.full_like(self.rstate._prev_step_mean,float('nan'))
        self.hupdates=t.zeros_like(self.lex_changes)
        self.original_observe=self.rstate.observe_sample
        self.rstate.observe_sample=self.observe_sample
        self.rstate._record_scales=self.record_harmonic

    def add_requests(self,output):
        super().add_requests(output)
        for r in output.scheduled_new_reqs:
            i=self.active[r.req_id]
            self.hready[i]=False;self.hprevious[i]=float('nan');self.hupdates[i]=0

    def accept_lexical_sample(self,idx,token,valid):
        # Native boundary/count rule includes all accepted non-boundary tokens;
        # do not silently add a different think-marker/semantic filter.
        s=self.rstate
        self.hready[idx]=valid & self.boundary[token] & (s._step_tok_count[idx]>0)
        self.hprevious[idx]=s._prev_step_mean[idx]
        super().accept_lexical_sample(idx,token,valid)

    def observe_sample(self,input_batch,sampled_token_ids,max_probabilities):
        # Saved RAW max probability arrives here after sampling, before the next
        # token's suppression. Native valid-prefill filtering remains in charge.
        inverse=max_probabilities.clamp_min(1e-12).reciprocal()
        return self.original_observe(input_batch,sampled_token_ids,inverse)

    def record_harmonic(self,batch,positions,pos,idx):
        t=self.torch;s=self.rstate
        params=s._dynamic_params[batch.req_ids[positions[0]]]
        if params.paper_parameters is not None:raise ValueError('Public-code controller only')
        ready=self.hready[idx]
        c=s._prev_step_mean[idx].clamp_min(1e-12).reciprocal()
        previous=self.hprevious[idx]
        v=t.where(t.isfinite(previous),(c-previous).square()/4,t.zeros_like(c))
        coef=self.compute_coefficient(c,v,params)
        s._coefs.index_copy_(0,idx,t.where(ready,coef,s._coefs[idx]))
        s._prev_step_mean.index_copy_(0,idx,t.where(ready,c,s._prev_step_mean[idx]))
        self.hupdates[idx]+=ready.long()
        self.original_record(batch,positions,pos,idx)

    def remove_request(self,rid):
        value=int(self.hupdates[self.active[rid]].cpu()) if rid in self.active else None
        result=super().remove_request(rid)
        if value is not None:self.completed[rid]['harmonic_step_updates']=value
        return result

    def close(self):
        if hasattr(self,'original_observe'):self.rstate.observe_sample=self.original_observe
        super().close()
