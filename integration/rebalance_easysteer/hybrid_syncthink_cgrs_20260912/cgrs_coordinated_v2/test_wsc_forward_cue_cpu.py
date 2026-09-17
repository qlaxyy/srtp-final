"""State, budget and accepted-token ordering checks; no model/GPU."""
from types import SimpleNamespace
import numpy as np
import torch
from wsc_forward_cue import CueState,ForcedSampler,WSCForwardCue
from wsc_feature_contract_cpu import BoundaryLedger


def main():
    WSCForwardCue(object(),None,None,None,None,enabled=False).close()
    state=CueState([7,8],cap=12)
    assert not state.observe(.9,20,8)
    assert state.observe(.9,20,9)
    assert state.first==9 and state.next_token(9)==7 and state.next_token(10)==8
    assert state.next_token(11) is None and not state.observe(.99,20,11)
    small=CueState([7,8],cap=10)
    small.observe(.9,20,8);assert small.observe(.9,20,9)
    assert small.skipped_budget and small.next_token(9) is None
    shadow=CueState([7],active=False)
    shadow.observe(.9,20,10);assert shadow.observe(.9,20,20)
    assert shadow.next_token(20) is None
    other=CueState([7,8]);assert other.first is None and other.loop.long==0
    ledger=BoundaryLedger(2,1536,{271});ledger.accept(271)
    event=ledger.observe(2,271,np.ones(1536));assert event['position']==0
    assert ledger.observe(1,5,None) is None
    ledger.accept(151649);assert ledger.observe(3,151649,None) is None and ledger.closed
    cue=CueState([7,8]);cue.observe(.9,10,0);cue.observe(.9,10,0)
    owner=SimpleNamespace(current_slots=[3],states={3:cue},ledgers={3:BoundaryLedger(2,1536,{271})})
    calls=[]
    def base(*a,**k):calls.append(1);return SimpleNamespace(sampled_token_ids=torch.tensor([[99]]))
    sampler=ForcedSampler(owner,base)
    result=sampler(None,SimpleNamespace(num_reqs=1))
    # Simulate RC14 and native R consumers AFTER the wrapped base sampler.
    seen_rc14=int(result.sampled_token_ids[0,0]);seen_native=int(result.sampled_token_ids[0,0])
    assert seen_rc14==seen_native==7 and len(calls)==1
    print('PASS: disabled identity, one shot, whole-cue budget, shadow no mutation, fresh state, processed boundary, accepted-token ordering')


if __name__=='__main__':main()
