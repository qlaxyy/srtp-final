"""Opt-in ablation: remove positive R coefficients before native history write."""
from label_alignment_adapter import AlignmentAdapter


def nonpositive(t,values):
    # Preserve negative and signed-zero values exactly.
    return t.where(values>0,t.zeros_like(values),values)


class NonpositiveAdapter(AlignmentAdapter):
    def __init__(self,*args,sign_mode='off',**kwargs):
        if sign_mode not in ('off','shadow','active'):raise ValueError(sign_mode)
        super().__init__(*args,**kwargs)
        self.sign_mode=sign_mode
        if not self.enabled or sign_mode=='off':return
        if self.lexical_control:raise ValueError('Do not combine label/control changes')
        t=self.torch;n=self.runner.max_num_reqs
        self.positive_seen=t.zeros(n,dtype=t.long,device=self.runner.device)
        self.sign_first=t.full_like(self.positive_seen,-1)
        self.sign_original_record=self.rstate._record_scales
        self.rstate._record_scales=self.record_nonpositive

    def record_nonpositive(self,batch,positions,pos,idx):
        t=self.torch;s=self.rstate;before=s._coefs[idx]
        t._assert_async(t.isfinite(before).all(),'Non-finite native R coefficient')
        positive=before>0
        self.positive_seen[idx]+=positive.long()
        # Sampler count already includes the just-accepted token. Changes act
        # on subsequent predictions, so tokens before this count must match.
        self.sign_first[idx]=t.where(positive&(self.sign_first[idx]<0),self.count[idx],self.sign_first[idx])
        if self.sign_mode=='active':s._coefs.index_copy_(0,idx,nonpositive(t,before))
        self.sign_original_record(batch,positions,pos,idx)

    def add_requests(self,output):
        super().add_requests(output)
        if self.sign_mode=='off':return
        for r in output.scheduled_new_reqs:
            i=self.active[r.req_id];self.positive_seen[i]=0;self.sign_first[i]=-1

    def remove_request(self,rid):
        extra=None
        if self.sign_mode!='off' and rid in self.active:
            i=self.active[rid];extra=self.torch.stack([self.positive_seen[i],self.sign_first[i]]).cpu().tolist()
        result=super().remove_request(rid)
        if extra is not None:self.completed[rid].update(positive_record_calls=extra[0],first_positive_change=extra[1],sign_mode=self.sign_mode)
        return result

    def close(self):
        if hasattr(self,'sign_original_record'):self.rstate._record_scales=self.sign_original_record
        super().close()


def cpu_checks():
    import torch
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        x=torch.tensor([-1.535345,-1.,-.5,-.001,-0.,0.,.00001,.01,.1],dtype=dtype)
        before=x.clone();y=nonpositive(torch,x)
        assert torch.equal(x,before) and torch.equal(y[x<=0],x[x<=0])
        assert torch.equal(torch.signbit(y[x==0]),torch.signbit(x[x==0]))
        assert (y[x>0]==0).all() and (y<=0).all()
        # Exercise actual hook order using a minimal fake native history sink.
        class State:pass
        class Native:pass
        for mode in ('shadow','active'):
            o=object.__new__(NonpositiveAdapter);o.torch=torch;o.sign_mode=mode
            o.rstate=State();o.rstate._coefs=x.clone();o.count=torch.arange(len(x))+10
            o.positive_seen=torch.zeros(len(x),dtype=torch.long);o.sign_first=torch.full_like(o.positive_seen,-1)
            seen=[];o.sign_original_record=lambda *a:seen.append(o.rstate._coefs.clone())
            idx=torch.tensor([1,4,7,8]);o.record_nonpositive(None,None,None,idx)
            expected=x.clone()
            if mode=='active':expected[idx]=nonpositive(torch,x[idx])
            assert torch.equal(seen[0],expected)
            assert o.sign_first[7]==17 and o.sign_first[8]==18 and o.sign_first[1]==-1
            assert torch.equal(o.rstate._coefs[[0,2,3,5,6]],x[[0,2,3,5,6]])
        checks.append(str(dtype)+' negative/zero/input preserved; actual hook writes history after change; shadow identical; unrelated slots untouched')
    return checks
