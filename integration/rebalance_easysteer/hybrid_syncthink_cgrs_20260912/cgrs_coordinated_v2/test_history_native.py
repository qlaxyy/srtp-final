"""CPU tensors in the existing Torch environment; no model or CUDA required."""
import unittest
import torch
from types import SimpleNamespace as N
from test_native import fixture
from adapter import Sampler
from policy import PENALTY
from replay_adapter import ReplayAdapter


class NativeHistoryTests(unittest.TestCase):
    def test_first_marker_then_later_mask(self):
        for dtype in (torch.float32,torch.bfloat16):
            o,b,x=fixture(dtype=dtype);o.history_gate='after_first_reflection'
            o.first_reflection=torch.full_like(o.count,-1)
            o.reflection_lookup=torch.zeros_like(o.clean);o.reflection_lookup[o.ids]=True
            o.original_sampler=lambda *a,**kw:N(sampled_token_ids=torch.tensor([[-1],[14190],[3983],[220]]))
            Sampler(o)(x,b);self.assertTrue(torch.equal(x,torch.ones_like(x)))
            self.assertEqual(o.first_reflection.tolist(),[20,-1,40,-1])
            o.opening[:]=True;b.seq_lens=torch.tensor([2,31,51,41])
            x=torch.ones_like(x);expected=x.clone();expected[1,o.ids]-=PENALTY
            Sampler(o)(x,b);self.assertTrue(torch.equal(x,expected))

    def test_actual_tensor_snapshot_and_slot_restore(self):
        o=ReplayAdapter.__new__(ReplayAdapter);o.torch=torch;o.history_gate='after_first_reflection'
        o.fields=ReplayAdapter.fields+('first_reflection',);o.active={'q':1};o.completed={}
        o.suspended={};o.suspending={'q'};o.replay_events=[];o.original_remove=lambda rid:None
        values=dict(opening=1,thinking=1,count=32,prompt_len=10,eligible_count=4,changed_count=4,first_change=15,first_reflection=7)
        for k,v in values.items():
            t=torch.zeros(4,dtype=torch.int64);t[1]=v;setattr(o,k,t)
        o.remove_request('q');self.assertEqual(o.suspended['q'],values)
        o.runner=N(req_states=N(req_id_to_index={'q':2}),steer_vector_state=N(_dynamic_indices={'q':2}))
        o.original_add=lambda output:None
        params=N(n=1,structured_outputs=None,logit_bias=None,bad_words=None,presence_penalty=0,frequency_penalty=0,repetition_penalty=1,min_tokens=0)
        r=N(req_id='q',sampling_params=params,num_computed_tokens=0,prompt_token_ids=[151648]*10,prefill_token_ids=[1]*42)
        o.add_requests(N(scheduled_new_reqs=[r]))
        self.assertEqual({k:int(getattr(o,k)[2]) for k in o.fields},values)


if __name__=='__main__':unittest.main(verbosity=2)
