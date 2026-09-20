"""CPU observer data contracts; cannot substitute for native GPU gate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from types import SimpleNamespace as NS
from observer import EndpointObserver,NativeAudit

def main():
    EndpointObserver(object(),enabled=False).close()
    x=EndpointObserver.__new__(EndpointObserver);x.pending={};x.torch=torch
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);r=torch.ones_like(a)
    x.before_steering(None,None,(a,r));saved=x.pending['pre'].clone()
    a.add_(2);assert torch.equal(saved,torch.tensor([[2.,3.],[4.,5.]],dtype=a.dtype))
    x.after_layer(None,None,(a,r));x.before_last(None,(None,a,r));assert x.pending['checked']
    assert torch.equal(x.pending['hidden']-x.pending['pre'],torch.full_like(a,2))
    try:x.before_last(None,(None,a+1,r))
    except RuntimeError:pass
    else:raise AssertionError('Residual mismatch undetected')
    state=NS(_coefs=torch.zeros(2),_prev_step_mean=torch.zeros(2),_step_tok_count=torch.zeros(2))
    def native(b,t,p):state._coefs.add_(1);state._prev_step_mean.copy_(p)
    state.observe_sample=native;runner=NS(steer_vector_state=state);audit=NativeAudit(runner,1)
    batch=NS(idx_mapping=torch.tensor([0,1]),num_reqs=2,num_draft_tokens=0)
    probs=torch.tensor([.3,.7]);tokens=torch.tensor([[4],[5]])
    state.observe_sample(batch,tokens,probs);probs.zero_();out=audit.export()
    assert abs(out[0][0,1]-.3)<1e-6 and out[0][0,2]==1
    try:state.observe_sample(batch,tokens,probs)
    except RuntimeError:pass
    else:raise AssertionError('Overflow undetected')
    audit.close();assert state.observe_sample is native
    print('PASS: disabled path, pre/post snapshot isolation, residual check, native pmax copy, overflow, restore')
if __name__=='__main__':main()
