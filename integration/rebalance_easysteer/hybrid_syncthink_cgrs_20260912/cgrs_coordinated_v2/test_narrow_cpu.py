import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

import numpy as np
from adapter import Adapter, Sampler
from policy import TRIGGERS, PENALTY, trigger_vocabulary
from replay_adapter import ReplayAdapter


class Tensor(np.ndarray):
    def long(self): return self.astype(np.int64)
    def to(self, dtype): return self.astype(dtype)
    def numel(self): return self.size
    @property
    def device(self): return 'cpu'


def tensor(x): return np.asarray(x).view(Tensor)


def sample(profile, mode='negative'):
    def check(x, message):
        if not bool(x): raise AssertionError(message)
    torch = N(as_tensor=lambda x, device=None: tensor(x), int64=np.int64,
              isfinite=np.isfinite, where=lambda c,x,y: tensor(np.where(c,x,y)),
              zeros_like=np.zeros_like, _assert_async=check)
    lookup = tensor(np.zeros(151936, dtype=bool))
    owner = N(torch=torch, mode=mode, ids=tensor(list(trigger_vocabulary(profile))),
        opening=tensor([True,True,True]), thinking=tensor([True,True,True]),
        count=tensor([1,1,0]), prompt_len=tensor([10,10,10]),
        eligible_count=tensor([0,0,0]), changed_count=tensor([0,0,0]),
        first_change=tensor([-1,-1,-1]), clean=lookup, boundary=lookup, white=lookup,
        runner=N(steer_vector_state=N(_coefs=tensor([-.5,.1,-.5]),
                                     _prev_step_mean=tensor([.95,.99,.95]))))
    calls=[]
    def native(logits, batch, **kwargs):
        calls.append(1)
        return N(sampled_token_ids=tensor(logits.argmax(axis=1)[:,None]))
    owner.original_sampler=native
    batch=N(num_draft_tokens=0,num_reqs=3,idx_mapping=tensor([0,1,2]),
            seq_lens=tensor([11,11,5]),num_computed_tokens_np=np.array([10,10,4]),
            num_scheduled_tokens=np.ones(3,dtype=int),prefill_len_np=np.array([10,10,10]))
    logits=tensor(np.zeros((3,151936),dtype=np.float32))
    logits[:,3983]=1.; logits[:,14190]=.9; logits[:,1000]=.8
    before=logits.copy(); result=Sampler(owner)(logits,batch)
    return before,logits,result.sampled_token_ids,calls,owner


class NarrowTests(unittest.TestCase):
    def test_fixed_exact_subset_and_default_copy(self):
        old=trigger_vocabulary(); new=trigger_vocabulary('narrow8')
        self.assertEqual(old,TRIGGERS); self.assertEqual(len(new),8)
        self.assertEqual(set(old)-set(new),{3983,1988,8088,714,75763,41109})
        self.assertEqual(set(new.values()),{'Wait',' Wait','wait',' wait',
            'Alternatively',' Alternatively','Hmm',' Hmm'})
        old.clear(); self.assertEqual(len(trigger_vocabulary()),14)
        with self.assertRaises(ValueError):trigger_vocabulary('typo')

    def test_disabled_profiles_install_nothing(self):
        class Trap:
            def __getattribute__(self, name):raise AssertionError(name)
        for cls in (Adapter,ReplayAdapter):
            for profile in ('original14','narrow8'):
                obj=cls(Trap(),Trap(),trigger_profile=profile)
                obj.close(); self.assertFalse(obj.enabled)

    def test_actual_sampler_only_releases_six_columns_at_eligible_row(self):
        before,old,old_tokens,calls,_=sample('original14')
        _,new,new_tokens,new_calls,_=sample('narrow8')
        changed=np.argwhere(old!=new)
        self.assertEqual(set(map(tuple,changed)),{(0,i) for i in (3983,1988,8088,714,75763,41109)})
        self.assertTrue(np.array_equal(old[1:],before[1:]))
        self.assertTrue(np.array_equal(new[1:],before[1:]))
        self.assertEqual(int(old_tokens[0,0]),1000)
        self.assertEqual(int(new_tokens[0,0]),3983)
        self.assertAlmostEqual(float(new[0,14190]),.9-PENALTY,places=6)
        self.assertEqual(len(calls),1);self.assertEqual(len(new_calls),1)

    def test_shadow_is_identical_for_both_profiles(self):
        for profile in ('original14','narrow8'):
            before,after,_,_,owner=sample(profile,'shadow')
            self.assertTrue(np.array_equal(before,after))
            self.assertEqual(owner.count.tolist(),[2,2,0])

    def test_no_new_sampler_or_replay_state(self):
        self.assertEqual(ReplayAdapter.fields,('opening','thinking','count','prompt_len',
            'eligible_count','changed_count','first_change'))
        tree=ast.parse((Path(__file__).parent/'adapter.py').read_text(encoding='utf8'))
        sampler=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sampler')
        calls=[n for n in ast.walk(sampler) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in
            {'cpu','item','tolist','numpy','generate','enqueue'} for n in calls))


if __name__=='__main__':unittest.main(verbosity=2)
