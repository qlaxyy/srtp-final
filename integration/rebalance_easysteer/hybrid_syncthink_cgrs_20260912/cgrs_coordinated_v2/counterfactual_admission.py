"""Explicit synchronous generated-prefix admission; no shared scheduler edit."""
from counterfactual_snapshot import identity,stage,discard_staged
from replay_label_alignment import ReplayAlignmentAdapter

def seed_request(scheduler,rid,snapshot,*,remaining_tokens,asset_identity):
    if snapshot['asset_identity']!=asset_identity:raise ValueError('Asset mismatch')
    if snapshot.get('action') not in ('apply','skip'):raise ValueError('Fork action required')
    req=scheduler.requests.get(rid)
    if req is None:raise ValueError('Request must be admitted before prefix seeding')
    if req in scheduler.running or req.num_computed_tokens or req.num_output_tokens:
        raise ValueError('Request already started')
    prompt=list(req.prompt_token_ids);generated=snapshot['generated_ids']
    if identity(prompt,generated)!=snapshot['prefix_sha256']:raise ValueError('Prompt differs')
    if not 0<remaining_tokens<=16000-len(generated):raise ValueError('Invalid total budget')
    if req.max_tokens!=len(generated)+remaining_tokens:raise ValueError('Scheduler must charge saved output prefix')
    if req.num_output_placeholders:raise ValueError('Async placeholders unsupported')
    if len(req.all_token_ids)!=len(prompt):raise ValueError('Unexpected initial sequence')
    req.append_output_token_ids(list(generated))
    if list(req.all_token_ids)!=prompt+generated:raise RuntimeError('Native append violated prefix identity')

class ForkReplayAdapter(ReplayAlignmentAdapter):
    def __init__(self,*args,asset_identity,**kwargs):
        self.pending_forks={};self.asset_identity=asset_identity;self.admitted_forks=[];self.prefill_masks={}
        super().__init__(*args,**kwargs)
        self.core.scheduler._preempt_request=self.reject_preempt
        from vllm.v1.worker.gpu import model_runner
        self.fill_module=model_runner;self.original_fill=model_runner.fill_graph_steer_buffers
        model_runner.fill_graph_steer_buffers=self.audit_fill

    def audit_fill(self,batch,state,manager):
        self.original_fill(batch,state,manager)
        entries=manager.graph_batch_entries()
        for b,rid in enumerate(batch.req_ids):
            if rid not in self.admitted_forks or rid in self.prefill_masks:continue
            length=state._history_lengths[rid]
            computed=int(batch.num_computed_tokens_np[b]);count=int(batch.num_scheduled_tokens[b])
            if computed!=0 or count!=length:raise RuntimeError('Fork gate requires one complete prefill partition')
            slot=state._slots[rid];controllers=entries[slot][2]
            if len(controllers)!=1:raise RuntimeError('One original vector required')
            start=int(batch.query_start_loc_np[b])
            self.prefill_masks[rid]=controllers[0].graph_mask[start:start+count].detach().float().cpu().tolist()

    def close(self):
        if hasattr(self,'original_fill'):self.fill_module.fill_graph_steer_buffers=self.original_fill
        super().close()

    def bind(self,rid,snapshot,*,remaining_tokens):
        if rid in self.pending_forks:raise ValueError('Fork already bound')
        seed_request(self.core.scheduler,rid,snapshot,remaining_tokens=remaining_tokens,asset_identity=self.asset_identity)
        self.pending_forks[rid]=snapshot

    def add_requests(self,output):
        staged=[]
        try:
            for req in output.scheduled_new_reqs:
                snapshot=self.pending_forks.get(req.req_id)
                if snapshot is not None:
                    stage(self,req.req_id,snapshot,req,asset_identity=self.asset_identity);staged.append(req.req_id)
            super().add_requests(output)
            for rid in staged:
                self.pending_forks.pop(rid);self.admitted_forks.append(rid)
        except BaseException:
            for rid in staged:
                if rid not in self.active and rid not in self.runner.steer_vector_state._dynamic_indices:discard_staged(self,rid)
            raise
