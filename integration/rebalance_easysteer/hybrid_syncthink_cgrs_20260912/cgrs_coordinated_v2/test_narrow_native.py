"""Run only in existing inference environment: CPU FP32/BF16, no model."""
import unittest
import torch
from adapter import Sampler
from policy import PENALTY,trigger_vocabulary
from test_native import fixture


class NativeNarrowTests(unittest.TestCase):
    def test_exact_released_columns_and_original_default(self):
        for dtype in (torch.float32,torch.bfloat16):
            for profile in ('original14','narrow8'):
                owner,batch,x=fixture(dtype=dtype)
                owner.ids=torch.tensor(list(trigger_vocabulary(profile)))
                expected=x.clone();expected[1,owner.ids]-=PENALTY
                Sampler(owner)(x,batch)
                self.assertTrue(torch.equal(x,expected))
                self.assertEqual(owner.count.tolist(),[21,31,41,50])
                if profile=='narrow8':
                    self.assertEqual(x[1,[3983,1988,8088,714,75763,41109]].tolist(),[1.]*6)


if __name__=='__main__':unittest.main(verbosity=2)
