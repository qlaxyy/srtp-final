"""Synchronous KV-eviction lifecycle for the existing lexical sampler.

Native ReBalance owns hidden-state history replay. No new controller or sampler.
"""
from adapter import Adapter


class ReplayAdapter(Adapter):
    fields=('opening','thinking','count','prompt_len','eligible_count','changed_count','first_change')

    def __init__(self,llm,tokenizer,mode='off',gate_on=False):
        super().__init__(llm,tokenizer,mode,gate_on)
        if not self.enabled:return
        if self.runner.vllm_config.scheduler_config.async_scheduling:
            super().close()
            raise ValueError('Native R KV history replay is only validated synchronously')
        self.suspended={};self.suspending=set();self.replay_events=[]
        self.original_finish=self.runner.finish_requests
        self.runner.finish_requests=self.finish_requests
        self.core.scheduler._preempt_request=self.original_preempt

    def finish_requests(self,output):
        self.suspending=set(output.preempted_req_ids or ())-set(output.finished_req_ids)
        try:self.original_finish(output)
        finally:self.suspending=set()
        for rid in output.finished_req_ids:self.suspended.pop(rid,None)

    def remove_request(self,rid):
        if rid in self.suspending and rid in self.active:
            slot=self.active.pop(rid)
            values=self.torch.stack([getattr(self,k)[slot].to(self.torch.int64) for k in self.fields]).cpu().tolist()
            self.suspended[rid]=dict(zip(self.fields,values))
            self.replay_events.append(dict(request_id=rid,event='suspend',sample_count=values[2]))
            return self.original_remove(rid)
        return super().remove_request(rid)

    def add_requests(self,output):
        for r in output.scheduled_new_reqs:
            p=r.sampling_params
            if (r.num_computed_tokens or p is None or p.n!=1 or p.structured_outputs is not None
                    or p.logit_bias or p.bad_words or p.presence_penalty or p.frequency_penalty
                    or p.repetition_penalty!=1 or p.min_tokens):
                raise ValueError('Unsupported replay request options')
            generated=len(r.prefill_token_ids)-len(r.prompt_token_ids)
            if r.req_id in self.suspended:
                saved=self.suspended[r.req_id]
                if generated!=saved['count'] or len(r.prompt_token_ids)!=saved['prompt_len']:
                    raise RuntimeError('Lexical replay prefix clock mismatch')
            elif generated:
                raise RuntimeError('Generated prefix lacks lexical state')
        self.original_add(output)
        for r in output.scheduled_new_reqs:
            rid=r.req_id
            if rid in self.active or rid in self.completed:raise RuntimeError('Request ID reused')
            slot=self.runner.req_states.req_id_to_index[rid]
            if rid not in self.runner.steer_vector_state._dynamic_indices:raise RuntimeError('R required')
            self.active[rid]=slot
            saved=self.suspended.pop(rid,None)
            if saved is not None:
                for key,value in saved.items():getattr(self,key)[slot]=value
                self.replay_events.append(dict(request_id=rid,event='restore',sample_count=saved['count'],
                                               replay_prefill_tokens=len(r.prefill_token_ids)))
            else:
                self.opening[slot]=False;self.thinking[slot]=151648 in r.prompt_token_ids
                self.count[slot]=self.eligible_count[slot]=self.changed_count[slot]=0
                self.first_change[slot]=-1;self.prompt_len[slot]=len(r.prompt_token_ids)

    def close(self):
        if self.enabled:self.runner.finish_requests=self.original_finish
        super().close()
