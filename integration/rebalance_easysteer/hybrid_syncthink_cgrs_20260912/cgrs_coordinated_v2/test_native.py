"""Existing inference environment, CPU Torch tensors; no model/GPU loading."""
from types import SimpleNamespace as N
import unittest
import torch
from adapter import Sampler
from policy import PENALTY, TRIGGERS


def fixture(mode='negative',dtype=torch.float32):
    size=151936
    o=N(torch=torch,mode=mode,ids=torch.tensor(list(TRIGGERS)),
        opening=torch.tensor([True,True,True,True]),thinking=torch.ones(4,dtype=torch.bool),
        count=torch.tensor([20,30,40,50]),eligible_count=torch.zeros(4,dtype=torch.long),
        prompt_len=torch.full((4,),10),
        changed_count=torch.zeros(4,dtype=torch.long),first_change=torch.full((4,),-1),
        clean=torch.zeros(size,dtype=torch.bool),white=torch.zeros(size,dtype=torch.bool),
        boundary=torch.zeros(size,dtype=torch.bool))
    o.clean[271]=o.boundary[271]=True
    o.white[220]=True
    o.runner=N(steer_vector_state=N(_coefs=torch.tensor([-.5,.1,-1.,-1.]),
        _prev_step_mean=torch.tensor([.95,.99,float('nan'),.98])))
    batch=N(num_draft_tokens=0,num_reqs=4,idx_mapping=torch.tensor([3,0,2,1]),
        seq_lens=torch.tensor([2,30,50,40]),
        num_computed_tokens_np=torch.tensor([1,9,9,9]).numpy(),
        num_scheduled_tokens=torch.ones(4,dtype=torch.int32).numpy(),
        prefill_len_np=torch.full((4,),10).numpy())
    out=N(sampled_token_ids=torch.tensor([[-1],[151649],[271],[220]]))
    o.original_sampler=lambda logits,batch,**kwargs:out
    return o,batch,torch.ones(4,size,dtype=dtype)


class TensorTests(unittest.TestCase):
    def test_routing_partial_prefill_and_closed_answer(self):
        for dtype in (torch.float32,torch.bfloat16):
            o,b,x=fixture(dtype=dtype);before=x.clone()
            Sampler(o)(x,b)
            expected=before.clone();expected[1,o.ids]-=PENALTY
            self.assertTrue(torch.equal(x,expected))
            self.assertEqual(o.changed_count.tolist(),[1,0,0,0])
            self.assertEqual(o.count.tolist(),[21,31,41,50])
            self.assertFalse(bool(o.thinking[0]))
            self.assertFalse(bool(o.opening[0]))
            self.assertTrue(bool(o.opening[3])) # dummy -1 did not corrupt slot3

    def test_shadow_is_byte_exact(self):
        o,b,x=fixture(mode='shadow',dtype=torch.bfloat16);before=x.clone()
        Sampler(o)(x,b)
        self.assertTrue(torch.equal(x,before))
        self.assertEqual(o.eligible_count.tolist(),[1,0,0,0])

    def test_lexical_only_control_needs_no_R_state(self):
        o,b,x=fixture(mode='always');o.runner.steer_vector_state=None
        Sampler(o)(x,b)
        self.assertEqual(o.changed_count.tolist(),[1,1,1,0])


if __name__=='__main__':
    unittest.main(verbosity=2)
