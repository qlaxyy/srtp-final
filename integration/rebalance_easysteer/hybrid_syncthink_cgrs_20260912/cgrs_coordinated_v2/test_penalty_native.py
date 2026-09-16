"""Native CPU Torch only; no CUDA/model initialization."""
import unittest
import torch
from adapter import Sampler
from test_native import fixture
from policy import PENALTY


class NativePenaltyTests(unittest.TestCase):
    def test_modes_fp32_bf16(self):
        for dtype in (torch.float32,torch.bfloat16):
            for mode in ('fixed','coefficient_scaled','calibration_constant'):
                o,b,x=fixture(dtype=dtype)
                o.penalty_mode=mode;o.lower_bound=-1.;o.constant_scale=.3
                scale={'fixed':1.,'coefficient_scaled':.5,'calibration_constant':.3}[mode]
                expected=x.clone()
                if mode=='fixed':expected[1,o.ids]-=PENALTY
                else:expected[1,o.ids]=(expected[1,o.ids].float()-PENALTY*scale).to(dtype)
                Sampler(o)(x,b)
                self.assertTrue(torch.equal(x,expected),(dtype,mode))

    def test_scaled_shadow_and_near_zero(self):
        for dtype in (torch.float32,torch.bfloat16):
            o,b,x=fixture(mode='shadow',dtype=dtype)
            o.penalty_mode='coefficient_scaled';o.lower_bound=-1.;o.constant_scale=None
            before=x.clone();Sampler(o)(x,b);self.assertTrue(torch.equal(x,before))
            o,b,x=fixture(dtype=dtype);o.penalty_mode='coefficient_scaled';o.lower_bound=-1.;o.constant_scale=None
            o.runner.steer_vector_state._coefs[0]=-1e-10
            before=x.clone();Sampler(o)(x,b);self.assertTrue(torch.equal(x,before))


if __name__=='__main__':unittest.main()
