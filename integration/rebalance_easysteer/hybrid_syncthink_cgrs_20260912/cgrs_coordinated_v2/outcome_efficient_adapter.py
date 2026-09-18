"""Opt-in one-sided efficient-endpoint control; preserve native confidence/history."""
from label_alignment_adapter import AlignmentAdapter


def efficient_coefficient(c,low,high):
    return -((high-c)/(high-low)).clamp(0,1)


class OutcomeEfficientAdapter(AlignmentAdapter):
    def __init__(self,*args,control_mode='off',**kwargs):
        if control_mode not in ('off','shadow','active'):raise ValueError(control_mode)
        self.control_mode=control_mode
        super().__init__(*args,**kwargs)
        if not self.enabled or control_mode=='off':return
        if not self.large_suppression or self.lexical_control:raise ValueError('Frozen L27 suppression required')
        self.endpoint_ready=self.torch.zeros_like(self.lex_ready)
        self.endpoint_updates=self.torch.zeros_like(self.lex_changes)
        self.endpoint_changes=self.torch.zeros_like(self.lex_changes)
        self.rstate._record_scales=self.record_endpoint

    def add_requests(self,output):
        super().add_requests(output)
        if self.control_mode=='off':return
        for r in output.scheduled_new_reqs:
            i=self.active[r.req_id];self.endpoint_ready[i]=False;self.endpoint_updates[i]=0;self.endpoint_changes[i]=0

    def accept_lexical_sample(self,idx,token,valid):
        if self.control_mode!='off':
            self.endpoint_ready[idx]=valid & self.boundary[token] & (self.rstate._step_tok_count[idx]>0)
        super().accept_lexical_sample(idx,token,valid)

    def record_endpoint(self,batch,positions,pos,idx):
        t=self.torch;s=self.rstate;params=s._dynamic_params[batch.req_ids[positions[0]]]
        ready=self.endpoint_ready[idx];c=s._prev_step_mean[idx]
        # Native mean is already computed from pre-suppression raw probabilities.
        value=efficient_coefficient(c.nan_to_num(params.q75c),params.q25c,params.q75c)
        before=s._coefs[idx];self.endpoint_updates[idx]+=ready.long()
        self.endpoint_changes[idx]+=(ready & (before!=value)).long()
        if self.control_mode=='active':s._coefs.index_copy_(0,idx,t.where(ready,value,before))
        self.original_record(batch,positions,pos,idx)

    def remove_request(self,rid):
        extra=None
        if self.control_mode!='off' and rid in self.active:
            i=self.active[rid];extra=self.torch.stack([self.endpoint_updates[i],self.endpoint_changes[i]]).cpu().tolist()
        result=super().remove_request(rid)
        if extra is not None:self.completed[rid].update(endpoint_updates=extra[0],endpoint_changes=extra[1],endpoint_mode=self.control_mode)
        return result


def cpu_checks():
    import torch
    c=torch.tensor([0.,.8,.9,1.]);y=efficient_coefficient(c,.8,.9)
    torch.testing.assert_close(y,torch.tensor([-1.,-1.,0.,0.]))
    grid=efficient_coefficient(torch.linspace(0,1,10001),.8,.9)
    assert (grid[1:]>=grid[:-1]).all() and grid.min()==-1 and grid.max()==0
    mo=torch.tensor([1.,3.]);me=torch.tensor([-2.,4.]);assert torch.equal(mo-(mo-me),me)
    from types import SimpleNamespace as N
    for mode in ['shadow','active']:
        obj=object.__new__(OutcomeEfficientAdapter);obj.torch=torch;obj.control_mode=mode
        obj.endpoint_ready=torch.tensor([True,False,True]);obj.endpoint_updates=torch.zeros(3,dtype=torch.long);obj.endpoint_changes=torch.zeros(3,dtype=torch.long)
        params=N(q25c=.8,q75c=.9)
        obj.rstate=N(_dynamic_params={'x':params},_prev_step_mean=torch.tensor([.8,.85,.9]),_coefs=torch.tensor([-.5,.123,.05]))
        seen=[];obj.original_record=lambda *args:seen.append(obj.rstate._coefs.clone())
        obj.record_endpoint(N(req_ids=['x']),[0],None,torch.arange(3))
        torch.testing.assert_close(seen[0],torch.tensor([-.5,.123,.05]) if mode=='shadow' else torch.tensor([-1.,.123,0.]))
        assert obj.endpoint_updates.tolist()==[1,0,1]
    return {'anchors':[-1,0],'finite_bounded_monotone':True,'target_centroid_exact':True,'shadow_and_history_order':True,'unready_slots_unchanged':True}
