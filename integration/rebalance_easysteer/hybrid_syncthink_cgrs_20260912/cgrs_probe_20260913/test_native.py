"""Run in existing inference environment before GPU engineering; CPU tensors only."""
import math
from types import SimpleNamespace
import unittest

import torch

from backend import SamplerAdapter, Backend
from policy import Config, State, TRIGGERS


class NativeTensorTests(unittest.TestCase):
    def test_bulk_signature_preserves_exact_scalar_and_history_bytes(self):
        owner = Backend.__new__(Backend)
        owner.torch = torch
        fields = [torch.tensor([float('nan'), .2]), torch.tensor([1, 2])]
        state = SimpleNamespace(_dynamic_indices={'a': 0, 'b': 1},
            _state_fields=lambda: fields, _history=torch.arange(16).reshape(2, 8),
            _history_lengths={'a': 3, 'b': 5})
        owner.runner = SimpleNamespace(steer_vector_state=state)
        requests = [SimpleNamespace(request_id=r, all_token_ids=[1,2],
                                    num_computed_tokens=1, num_output_tokens=1) for r in ('b','a')]
        old = owner.primary_signature(requests)
        owner.bulk_signature = True
        self.assertEqual(old, owner.primary_signature(requests))

    def test_cache_copy_after_zeroing_is_disjoint_and_byte_exact(self):
        owner = Backend.__new__(Backend)
        owner.torch = torch
        cache = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
        owner.runner = SimpleNamespace(kv_caches=[cache],
                                        kv_cache_config=SimpleNamespace(num_blocks=4))
        parent = SimpleNamespace(num_computed_tokens=2, request_id='main',
                                 all_token_ids=[1, 2, 3])
        owner.scheduler = SimpleNamespace(requests={'main': parent},
            kv_cache_manager=SimpleNamespace(get_block_ids=lambda rid: ([0],)))
        owner.probes = {'child': dict(parent='main', cached_tokens=2)}
        owner.block_size, owner.verify_cache = 2, True
        owner.clone_checks, owner.clone_bytes = [], 0
        owner.cache_views = owner.cache_block_views()
        owner.original_update = lambda output: cache[2].zero_()
        child = SimpleNamespace(req_id='child', num_computed_tokens=2,
                                 prompt_token_ids=[1, 2, 3, 4], block_ids=([2],))
        owner.update_requests(SimpleNamespace(scheduled_new_reqs=[child]))
        self.assertTrue(torch.equal(cache[0], cache[2]))
        self.assertTrue(owner.clone_checks[0]['byte_equal'])
        self.assertEqual(owner.clone_bytes, 16)
        original = cache[0].clone()
        cache[2].fill_(255)
        self.assertTrue(torch.equal(cache[0], original))
        child.block_ids = ([0],)
        with self.assertRaisesRegex(RuntimeError, 'disjoint'):
            owner.update_requests(SimpleNamespace(scheduled_new_reqs=[child]))
        child.block_ids = ([2],)
        child.prompt_token_ids[0] = 99
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            owner.update_requests(SimpleNamespace(scheduled_new_reqs=[child]))

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
