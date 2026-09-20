"""CPU lifecycle check: preserve every lexical field across a slot change."""
from types import SimpleNamespace as NS
import torch
from replay_label_alignment import ReplayAlignmentAdapter as A

def main():
    a=A.__new__(A);a.torch=torch;a.active={'r':0};a.completed={};a.suspended={};a.suspending={'r'};a.replay_events=[]
    for k in A.fields:setattr(a,k,torch.zeros(2,dtype=torch.int64))
    for i,k in enumerate(A.fields):getattr(a,k)[0]=i+1
    a.count[0]=3;a.prompt_len[0]=2;a.lex_state[0]=117;a.lex_open[0]=1
    before={k:int(getattr(a,k)[0]) for k in A.fields}
    a.original_remove=lambda rid:None
    a.remove_request('r')
    assert a.suspended['r']==before and not a.active and not a.completed
    for k in A.fields:getattr(a,k).fill_(-9)
    a.suspending=set();a.original_add=lambda output:None
    a.runner=NS(req_states=NS(req_id_to_index={'r':1}),steer_vector_state=NS(_dynamic_indices={'r':1}))
    sp=NS(n=1,structured_outputs=None,logit_bias=None,bad_words=None,presence_penalty=0,frequency_penalty=0,repetition_penalty=1,min_tokens=0)
    req=NS(req_id='r',num_computed_tokens=0,sampling_params=sp,prefill_token_ids=[1,2,3,4,5],prompt_token_ids=[1,2])
    a.add_requests(NS(scheduled_new_reqs=[req]))
    assert {k:int(getattr(a,k)[1]) for k in A.fields}==before
    assert not a.suspended and a.active=={'r':1}
    a.suspending={'r'};a.remove_request('r');req.prefill_token_ids.append(6)
    try:a.add_requests(NS(scheduled_new_reqs=[req]))
    except RuntimeError as e:assert 'clock mismatch' in str(e)
    else:raise AssertionError('Mismatched replay accepted')
    print('PASS: all 13 state fields restored after slot relocation; bad prefix clock rejected')
if __name__=='__main__':main()
