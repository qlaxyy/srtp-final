"""Opt-in L27 margin protection; no extra model call or recurrent policy state."""
import math
from label_alignment_adapter import AlignmentAdapter


def device_adjust(t, logits, ids, mask):
    values=logits[:,ids]
    active=mask & t.isfinite(values)
    reflected=t.where(active,values,-t.inf).float().amax(-1)
    alternatives=logits.float().clone()
    alternatives[:,ids]=t.where(active,-t.inf,alternatives[:,ids])
    other=alternatives.amax(-1)
    valid=t.isfinite(reflected)&t.isfinite(other)
    gap=t.where(valid,(reflected-other).clamp_min(0),0.)
    amount=t.where(valid,(math.log(2)-gap).clamp(0,math.log(2)),0.)
    fixed=values-math.log(2)
    adjusted=(values.float()-amount[:,None]).to(values.dtype)
    adjusted=t.where((gap==0)[:,None]&valid[:,None],fixed,adjusted)
    adjusted=t.where((amount==0)[:,None],values,adjusted)
    return t.where(active,adjusted,values)


class MarginAdapter(AlignmentAdapter):
    def __init__(self,*args,margin_mode='off',**kwargs):
        if margin_mode not in ('off','shadow','active'):raise ValueError(margin_mode)
        super().__init__(*args,**kwargs)
        self.margin_mode=margin_mode
        if margin_mode=='off':return
        t=self.torch;n=self.runner.max_num_reqs
        self.margin_count=t.zeros(n,dtype=t.long,device=self.runner.device)
        self.margin_first=t.full_like(self.margin_count,-1)
        self.penalty_provider=self.adjust

    def adjust(self,logits,mask,idx):
        t=self.torch
        t._assert_async((~t.isnan(logits)&~t.isposinf(logits)).all(),'Invalid margin input logits')
        fixed=logits[:,self.ids]-math.log(2)
        adjusted=device_adjust(t,logits,self.ids,mask)
        changed=(mask & (adjusted!=fixed)).any(-1)
        self.margin_count[idx]+=changed.long()
        self.margin_first[idx]=t.where(changed & (self.margin_first[idx]<0),self.count[idx],self.margin_first[idx])
        return fixed if self.margin_mode=='shadow' else adjusted

    def add_requests(self,output):
        super().add_requests(output)
        if self.margin_mode=='off':return
        for r in output.scheduled_new_reqs:
            i=self.active[r.req_id];self.margin_count[i]=0;self.margin_first[i]=-1

    def remove_request(self,rid):
        extra=None
        if self.margin_mode!='off' and rid in self.active:
            i=self.active[rid]
            extra=self.torch.stack([self.margin_count[i],self.margin_first[i]]).cpu().tolist()
        result=super().remove_request(rid)
        if extra is not None:
            self.completed[rid].update(margin_changes=extra[0],margin_first=extra[1])
        return result
