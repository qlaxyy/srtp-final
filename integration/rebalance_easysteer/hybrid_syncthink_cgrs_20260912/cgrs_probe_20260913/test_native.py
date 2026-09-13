"""Run in existing inference environment before GPU engineering; CPU tensors only."""
import math
from types import SimpleNamespace
import unittest

import torch

from backend import SamplerAdapter
from policy import Config, State, TRIGGERS


class NativeTensorTests(unittest.TestCase):
    def test_raw_full_entropy_before_temperature_and_filter(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.zeros((1, 151936), dtype=dtype)
            probe = dict(audit=False, entropy=[])
            owner = SimpleNamespace(torch=torch, probes={'p': probe},
                                    expected_logits={}, states={}, suppress=False,
                                    callback_host_seconds=0.)
            batch = SimpleNamespace(num_draft_tokens=0, num_reqs=1, req_ids=['p'],
                                    num_computed_tokens_np=[9], num_scheduled_tokens=[1],
                                    prefill_len_np=[10])
            def sampler(x, batch):
                self.assertIs(x, logits)
                return 'unchanged'
            self.assertEqual(SamplerAdapter(owner, sampler)(logits, batch), 'unchanged')
            self.assertAlmostEqual(probe['entropy'][0], math.log(151936), places=5)
            self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_native_bf16_mask_does_not_touch_other_rows(self):
        s = State('a', Config())
        s.opening, s.p = True, 1.
        owner = SimpleNamespace(torch=torch, probes={}, expected_logits={},
                                states={'a': s}, suppress=True, callback_host_seconds=0.)
        batch = SimpleNamespace(num_draft_tokens=0, num_reqs=2, req_ids=['a', 'b'],
                                num_computed_tokens_np=[9, 9], num_scheduled_tokens=[1, 1],
                                prefill_len_np=[10, 10])
        logits = torch.ones((2, 151936), dtype=torch.bfloat16)
        SamplerAdapter(owner, lambda x, b: x)(logits, batch)
        self.assertEqual(int(torch.isneginf(logits[0]).sum()), len(TRIGGERS))
        self.assertTrue(torch.equal(logits[1], torch.ones_like(logits[1])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
