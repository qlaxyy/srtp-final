"""Bounded synchronous engineering observer; no intervention or KV mutation."""
from replay_label_alignment import ReplayAlignmentAdapter
from counterfactual_snapshot import capture

class SnapshotObserver(ReplayAlignmentAdapter):
    def __init__(self,*args,asset_identity,**kwargs):
        self.prefixes={};self.generated={};self.snapshots={};self.asset_identity=asset_identity
        super().__init__(*args,**kwargs)
        if not self.enabled:raise ValueError('Observer requires explicit enabled mode')
        self.saved_record=self.rstate._record_scales
        self.rstate._record_scales=self.record_snapshot
        # This batch validates capture, not unplanned eviction during capture.
        self.core.scheduler._preempt_request=self.reject_preempt

    def add_requests(self,output):
        super().add_requests(output)
        for r in output.scheduled_new_reqs:
            if r.req_id in self.prefixes:raise ValueError('Unexpected resume in capture-only batch')
            self.prefixes[r.req_id]=list(r.prompt_token_ids);self.generated[r.req_id]=[]

    def accept_lexical_sample(self,idx,token,valid):
        super().accept_lexical_sample(idx,token,valid)
        inverse={slot:rid for rid,slot in self.active.items()}
        # Engineering-only host synchronization, explicitly charged in timing.
        for slot,value,accepted in zip(idx.cpu().tolist(),token.cpu().tolist(),valid.cpu().tolist()):
            if accepted:self.generated[inverse[slot]].append(value)

    def record_snapshot(self,batch,positions,pos,idx):
        self.saved_record(batch,positions,pos,idx)
        for p,i in zip(positions,idx.cpu().tolist()):
            rid=batch.req_ids[p];ids=self.generated[rid]
            if rid in self.snapshots or len(ids)<64:continue
            s=self.rstate;params=s._dynamic_params[rid]
            if (ids[-1] not in params.boundary_token_ids or not bool(s._in_think[i])
                    or int(s._step_tok_count[i])!=0 or not self.torch.isfinite(s._prev_step_mean[i])):continue
            self.snapshots[rid]=capture(self,rid,self.prefixes[rid],ids,asset_identity=self.asset_identity)

    def close(self):
        if hasattr(self,'saved_record'):self.rstate._record_scales=self.saved_record
        super().close()
