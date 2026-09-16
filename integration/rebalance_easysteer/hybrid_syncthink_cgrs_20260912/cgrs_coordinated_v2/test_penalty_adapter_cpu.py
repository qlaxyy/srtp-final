import math
from types import SimpleNamespace as N
import unittest
import numpy as np
from adapter import Adapter
from replay_adapter import ReplayAdapter
from policy import validate_penalty, device_penalty, PENALTY
from test_history_cpu import owner, tick
from test_narrow_cpu import tensor
from test_replay_cpu import fixture
from calibrate_penalty_scale_cpu import collect_updates, replay


def configured(mode='coefficient_scaled'):
    o=owner();o.history_gate='none';o.penalty_mode=mode;o.lower_bound=-1.;o.constant_scale=None
    o.torch.clamp=lambda x,min,max:tensor(np.clip(x,min,max))
    o.torch.full_like=lambda x,v:tensor(np.full_like(x,v))
    return o


class PenaltyAdapterTests(unittest.TestCase):
    def test_scaled_real_sampler_and_masks(self):
        o=configured();x=tick(o,[1000,1000,-1])
        self.assertAlmostEqual(float(x[0,14190]),1-PENALTY*.5,places=6)
        self.assertTrue(np.all(x[1:]==1))
        self.assertEqual(o.changed_count.tolist(),[1,0,0])

    def test_constant_and_shadow(self):
        o=configured('calibration_constant');o.constant_scale=.3
        self.assertAlmostEqual(float(tick(o,[1000,1000,-1])[0,14190]),1-PENALTY*.3,places=6)
        o=configured();o.mode='shadow'
        self.assertTrue(np.all(tick(o,[1000,1000,-1])==1))

    def test_zero_nan_and_answer_protection(self):
        for value in [0.,float('nan'),float('inf'),-float('inf')]:
            o=configured();o.runner.steer_vector_state._coefs[0]=value
            self.assertTrue(np.all(tick(o,[1000,1000,-1])==1))
        o=configured();o.thinking[0]=False
        self.assertTrue(np.all(tick(o,[1000,1000,-1])==1))

    def test_off_does_not_access_engine_or_validate_new_options(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError(name)
        for cls in (Adapter,ReplayAdapter):
            obj=cls(Trap(),Trap(),penalty_mode='coefficient_scaled');obj.close()

    def test_configuration_and_bound_match(self):
        for args in [('fixed',-1,None),('coefficient_scaled',None,None),('coefficient_scaled',-1,.5),('calibration_constant',-1,0)]:
            with self.assertRaises(ValueError):validate_penalty(*args)
        validate_penalty('coefficient_scaled',-1,None)
        o=Adapter.__new__(Adapter);o.penalty_mode='coefficient_scaled';o.lower_bound=-1
        o.runner=N(steer_vector_state=N(_dynamic_params={'r':N(low_val_1=-.5,low_val_2=-1,paper_parameters=None)}))
        o.check_penalty_request('r');o.lower_bound=-2
        with self.assertRaises(ValueError):o.check_penalty_request('r')

    def test_replay_reuses_immutable_penalty_and_resets_existing_state(self):
        o,out,saved=fixture();rid=out.scheduled_new_reqs[0].req_id
        o.penalty_mode='coefficient_scaled';o.lower_bound=-1.;o.constant_scale=None
        o.runner.steer_vector_state._dynamic_params={rid:N(low_val_1=-.5,low_val_2=-1,paper_parameters=None)}
        o.add_requests(out);self.assertEqual(o.count[3],saved['count'])
        self.assertEqual(o.lower_bound,-1.)
        o.active={};o.suspended={};out.scheduled_new_reqs[0].prefill_token_ids=[1]*10
        o.add_requests(out);self.assertEqual(o.count[3],0);self.assertFalse(o.opening[3])

    def test_offline_empty_boundary_whitespace_and_causal_order(self):
        # First boundary closes a step; another boundary does not replace its R state.
        row={'prompt_token_ids':[151648], 'token_ids':[1,2,2,3,4,151649],
             'logprobs':[math.log(.8)]*6}
        boundaries={2};pieces={1:'work',2:'\n\n',3:' ',4:'Wait',151649:'</think>'}
        updates=collect_updates(row,boundaries)
        self.assertEqual(len(updates),1);self.assertAlmostEqual(updates[0][1],.8)
        scales,positions=replay(row,pieces,boundaries,updates,[-.5],-1)
        self.assertEqual(positions,[2,3,4]);self.assertEqual(scales,[.5]*3)


if __name__=='__main__':unittest.main()
