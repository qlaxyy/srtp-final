import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import numpy as np
import test_narrow_cpu as base
from adapter import Adapter,Sampler
from replay_adapter import ReplayAdapter
from test_replay_cpu import fixture as replay_fixture
from policy import PENALTY,TRIGGERS


def owner():
    o=base.sample('original14','shadow')[-1]
    o.mode='negative';o.history_gate='after_first_reflection'
    o.opening[:]=True;o.thinking[:]=True
    o.count[:]=0;o.first_reflection=base.tensor([-1,-1,-1])
    o.reflection_lookup=base.tensor(np.zeros(151936,dtype=bool))
    o.reflection_lookup[list(TRIGGERS)]=True
    o.clean=base.tensor(np.zeros(151936,dtype=bool));o.clean[271]=True
    o.boundary=o.clean.copy();o.boundary[1001]=True
    o.white=base.tensor(np.zeros(151936,dtype=bool));o.white[220]=True
    o.eligible_count[:]=0;o.changed_count[:]=0;o.first_change[:]=-1
    return o


def tick(o,tokens):
    b=N(num_draft_tokens=0,num_reqs=3,idx_mapping=base.tensor([0,1,2]),
        seq_lens=o.prompt_len+o.count,num_computed_tokens_np=np.array([10,10,0]),
        num_scheduled_tokens=np.ones(3,dtype=int),prefill_len_np=np.array([10,10,10]))
    o.original_sampler=lambda logits,batch,**kw:N(sampled_token_ids=base.tensor(np.array(tokens)[:,None]))
    x=base.tensor(np.ones((3,151936),dtype=np.float32));Sampler(o)(x,b)
    return x


class HistoryTests(unittest.TestCase):
    def test_first_opening_marker_is_free_later_negative_step_is_penalized(self):
        o=owner();x=tick(o,[14190,3983,-1])
        self.assertTrue(np.all(x==1));self.assertEqual(o.first_reflection.tolist(),[0,0,-1])
        self.assertTrue(np.all(tick(o,[1000,1000,-1])==1))
        tick(o,[271,271,-1]);x=tick(o,[14190,14190,-1])
        self.assertAlmostEqual(float(x[0,14190]),1-PENALTY,places=6)
        self.assertTrue(np.all(x[1:]==1));self.assertEqual(o.first_reflection.tolist(),[0,0,-1])

    def test_whitespace_preserves_permission_and_interior_marker_does_not_activate(self):
        o=owner();tick(o,[220,1000,-1]);tick(o,[14190,14190,-1])
        self.assertEqual(o.first_reflection.tolist(),[1,-1,-1])

    def test_mixed_boundary_and_answer_phase_do_not_activate(self):
        o=owner();tick(o,[1001,151649,-1]);tick(o,[14190,14190,-1])
        self.assertEqual(o.first_reflection.tolist(),[-1,-1,-1])

    def test_shadow_observes_but_changes_no_logits(self):
        o=owner();o.mode='shadow';tick(o,[14190,14190,-1]);tick(o,[271,271,-1])
        self.assertTrue(np.all(tick(o,[14190,14190,-1])==1))

    def test_default_matches_frozen_sampler_for_both_profiles(self):
        root=Path(__file__).resolve().parents[4]
        rel=Path(__file__).with_name('adapter.py').resolve().relative_to(root).as_posix()
        scope={};exec(compile(subprocess.check_output(['git','-C',str(root),'show','73cef5f:'+rel],text=True,encoding='utf8'),rel,'exec'),scope)
        for mode in ('negative','shadow'):
            for profile in ('original14','narrow8'):
                now=base.sample(profile,mode)
                with patch.object(base,'Sampler',scope['Sampler']):old=base.sample(profile,mode)
                for i in (1,2):self.assertTrue(np.array_equal(now[i],old[i]))
                for k in ReplayAdapter.fields:self.assertTrue(np.array_equal(getattr(now[-1],k),getattr(old[-1],k)))

    def test_disabled_installs_nothing(self):
        class Trap:
            def __getattribute__(self,n):raise AssertionError(n)
        for cls in (Adapter,ReplayAdapter):
            o=cls(Trap(),Trap(),history_gate='after_first_reflection');o.close()

    def test_replay_preserves_first_marker_and_fresh_slot_resets(self):
        o,out,saved=replay_fixture();o.history_gate='after_first_reflection'
        o.fields=ReplayAdapter.fields+('first_reflection',);o.first_reflection=np.full(4,-1)
        saved['first_reflection']=7;o.add_requests(out);self.assertEqual(o.first_reflection[3],7)
        o.active={};o.suspended={};out.scheduled_new_reqs[0].prefill_token_ids=[1]*10
        o.add_requests(out);self.assertEqual(o.first_reflection[3],-1)

    def test_missing_or_future_replay_position_rejected_before_native_mutation(self):
        for position in (None,32,-2):
            o,out,saved=replay_fixture();o.history_gate='after_first_reflection'
            o.fields=ReplayAdapter.fields+('first_reflection',);o.first_reflection=np.full(4,-1)
            if position is not None:saved['first_reflection']=position
            with self.assertRaisesRegex(RuntimeError,'reflection-history'):o.add_requests(out)
            self.assertFalse(o.active)

    def test_engineering_plan_cannot_be_used_for_formal_screening(self):
        from narrow_runner import cases
        p={'candidate_kind':'first_reflection'}
        self.assertEqual(len(cases(p,'engineering')),6)
        with self.assertRaises(AssertionError):cases(p,'screen')
        with self.assertRaises(AssertionError):cases(p,'speed')


if __name__=='__main__':unittest.main(verbosity=2)
