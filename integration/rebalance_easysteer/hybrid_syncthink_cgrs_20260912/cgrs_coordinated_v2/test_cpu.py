import ast
import math
from pathlib import Path
import unittest

from adapter import Adapter
from policy import eligible, next_opening, PENALTY, token_flags
from adapter import Sampler


class ContractTests(unittest.TestCase):
    def test_disabled_does_not_touch_engine_or_import_torch(self):
        class Trap:
            def __getattribute__(self,name):
                raise AssertionError('off touched engine')
        a = Adapter(Trap(), Trap())
        a.close()
        self.assertFalse(a.enabled)

    def test_negative_signal_and_valid_closed_step_required(self):
        for coefficient in (0,.1,float('nan'),float('inf')):
            self.assertFalse(eligible(True,True,coefficient,.99,'negative'))
        self.assertFalse(eligible(True,True,-1,float('nan'),'negative'))
        self.assertTrue(eligible(True,True,-.5,.95,'negative'))

    def test_content_and_answer_phase_are_protected(self):
        self.assertFalse(eligible(False,True,-1,.99,'negative'))
        self.assertFalse(eligible(True,False,-1,.99,'negative'))
        self.assertFalse(next_opening(True,'</think>',False,151649))

    def test_whitespace_and_mixed_boundary(self):
        self.assertTrue(next_opening(False,'\n\n',True,271))
        self.assertTrue(next_opening(True,' ',False,220))
        self.assertFalse(next_opening(True,'\n\nTherefore',True,999))
        self.assertFalse(next_opening(True,'Let',False,10))
        self.assertEqual(token_flags('x\n\n',True),(True,False))

    def test_soft_odds_and_temperature_are_distinct(self):
        # Two-token distribution: raw odds halve; sampling temperature changes
        # the factor, and neither means that normalized probability halves.
        self.assertAlmostEqual(math.exp(-PENALTY),.5)
        probability = math.exp(-PENALTY)/(1+math.exp(-PENALTY))
        self.assertAlmostEqual(probability,1/3)
        self.assertNotEqual(probability,.25)
        self.assertAlmostEqual(math.exp(-PENALTY/.7),.5**(1/.7))

    def test_shadow_uses_same_gate_without_defining_a_new_threshold(self):
        for c in (-1,-.5,0,.1):
            self.assertEqual(eligible(True,True,c,.95,'negative'),eligible(True,True,c,.95,'shadow'))

    def test_sampler_hot_path_has_no_host_reads_or_extra_sampling(self):
        tree=ast.parse((Path(__file__).parent/'adapter.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sampler')
        calls=[n for n in ast.walk(cls) if isinstance(n,ast.Call)]
        forbidden={'cpu','item','tolist','numpy','synchronize','generate','enqueue'}
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='original_sampler' for n in calls),1)

    def test_actual_sampler_with_numpy_tensor_standin(self):
        # Runs the actual adapter arithmetic; this does not emulate CUDA streams
        # or prove PyTorch BF16 equivalence. Native tests cover those types later.
        import numpy as np
        from types import SimpleNamespace as N
        class A(np.ndarray):
            def long(self):return self.astype(np.int64)
            def to(self,dtype):return self.astype(dtype)
            def numel(self):return self.size
            @property
            def device(self):return 'cpu'
        def a(value):return np.array(value).view(A)
        def check(value,message):
            if not bool(value):raise AssertionError(message)
        t=N(as_tensor=lambda x,device=None:a(x),int64=np.int64,
            isfinite=np.isfinite,where=lambda c,x,y:np.where(c,x,y).view(A),
            zeros_like=np.zeros_like,_assert_async=check)
        for mode in ('negative','shadow','always'):
            lookup=a([False]*151936);boundary=lookup.copy();boundary[271]=True
            o=N(torch=t,mode=mode,ids=a([1,2]),opening=a([True,True,True]),
                thinking=a([True,True,True]),count=a([1,2,3]),prompt_len=a([10,10,10]),
                eligible_count=a([0,0,0]),changed_count=a([0,0,0]),first_change=a([-1,-1,-1]),
                clean=boundary.copy(),boundary=boundary,white=lookup.copy(),
                runner=N(steer_vector_state=N(_coefs=a([-.5,.1,-1]),_prev_step_mean=a([.95,.99,.96]))))
            b=N(num_reqs=3,num_draft_tokens=0,idx_mapping=a([1,0,2]),seq_lens=a([12,11,2]),
                num_computed_tokens_np=np.array([11,10,1]),num_scheduled_tokens=np.ones(3,dtype=int),
                prefill_len_np=np.array([10,10,10]))
            o.original_sampler=lambda x,b:N(sampled_token_ids=a([[271],[151649],[-1]]))
            x=a(np.ones((3,3),dtype=np.float32));before=x.copy()
            Sampler(o)(x,b)
            expected=before.copy()
            if mode!='shadow':expected[1,1:]-=PENALTY
            if mode=='always':expected[0,1:]-=PENALTY
            self.assertTrue(np.array_equal(x,expected))
            self.assertEqual(o.count.tolist(),[2,3,3])
            self.assertFalse(o.thinking[0])
            self.assertTrue(o.opening[2])


if __name__=='__main__':
    unittest.main(verbosity=2)
