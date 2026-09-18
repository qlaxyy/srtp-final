"""CPU contract checks; no claim of vLLM scheduling or GPU equivalence."""
from types import SimpleNamespace as NS
import torch
from counterfactual_snapshot import FIELDS,capture,fork,stage,discard_staged
from replay_label_alignment import ReplayAlignmentAdapter

def fixture():
    s=NS(_dynamic_indices={'parent':0},_dynamic_params={'parent':NS(boundary_token_ids=[9])},
         _history_lengths={'parent':5},_prompt_lengths={'parent':2},_suspended={},supports_kv_replay=True)
    vals=[-.54321,0.,0,.923456789,True,0.,True]
    for name,value in zip(FIELDS,vals):setattr(s,name,torch.tensor([value,value]))
    s._state_fields=lambda:tuple(getattr(s,n) for n in FIELDS)
    s._history=torch.tensor([[0.,-1.,-.2,-.3,float(s._coefs[0])],[0.,0.,0.,0.,0.]])
    a=NS(torch=torch,fields=ReplayAlignmentAdapter.fields,active={'parent':0},completed={},suspended={},
         runner=NS(steer_vector_state=s,vllm_config=NS(scheduler_config=NS(async_scheduling=False))))
    for k in a.fields:setattr(a,k,torch.tensor([0,0]))
    a.count[0]=3;a.prompt_len[0]=2;a.thinking[0]=1;a.lex_open[0]=1;a.lex_state[0]=17
    return a

def main():
    a=fixture();s=a.runner.steer_vector_state;assets={'model':'hash-m','vector':'hash-v','runtime':'hash-r'}
    snap=capture(a,'parent',[1,2],[3,4,9],asset_identity=assets)
    original=s._history.clone();left=fork(snap,action='apply');right=fork(snap,action='skip')
    assert torch.equal(s._history,original)
    assert torch.equal(left['native']['history'][:-1],right['native']['history'][:-1])
    assert right['native']['history'][-1]==0 and left['native']['history'][-1]!=0
    for x,y in zip(left['native']['fields'],right['native']['fields']):assert torch.equal(x,y)
    assert left['lexical']==right['lexical'] and left['native']['fields'][0].item()==s._coefs[0].item()
    for rid,child in [('left',left),('right',right)]:
        req=NS(req_id=rid,num_computed_tokens=0,prompt_token_ids=[1,2],prefill_token_ids=[1,2,3,4,9])
        stage(a,rid,child,req,asset_identity=assets)
    s._suspended['right']['fields'][0].fill_(55)
    assert s._suspended['left']['fields'][0].item()==s._coefs[0].item()
    assert right['native']['fields'][0].item()==s._coefs[0].item()
    checks=3
    for change in ('token','prompt','assets','duplicate','async'):
        req=NS(req_id='bad',num_computed_tokens=0,prompt_token_ids=[1,2],prefill_token_ids=[1,2,3,4,9]);ident=assets
        if change=='token':req.prefill_token_ids[-1]=8
        if change=='prompt':req.prompt_token_ids=list(req.prefill_token_ids)
        if change=='assets':ident={'model':'different'}
        if change=='duplicate':req.req_id='left'
        if change=='async':a.runner.vllm_config.scheduler_config.async_scheduling=True
        before=set(a.suspended)
        try:stage(a,req.req_id,left,req,asset_identity=ident)
        except ValueError:pass
        else:raise AssertionError(change)
        assert set(a.suspended)==before;checks+=1
        a.runner.vllm_config.scheduler_config.async_scheduling=False
    discard_staged(a,'left');discard_staged(a,'right');assert not a.suspended and not s._suspended
    a.count[0]=2
    try:capture(a,'parent',[1,2],[3,4,9],asset_identity=assets)
    except ValueError:pass
    else:raise AssertionError('clock accepted')
    print('PASS',checks+2,'CPU invariants; GPU/scheduler replay NOT validated')

if __name__=='__main__':main()
