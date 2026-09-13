"""CPU contract tests: policy, isolation, raw sampling, and paused lifecycle.

Native torch/vLLM integration is a separate mandatory gate, not mocked as passed.
"""
from contextlib import ExitStack
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from policy import Config, State, TRIGGERS, boxed_certainty, draw, probability
from backend import Backend, SamplerAdapter, parked


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


class PolicyTests(unittest.TestCase):
    def test_full_entropy_uniform_and_certain_probes(self):
        tok = Tokenizer(['{', '12', '}'])
        self.assertEqual(boxed_certainty([0, 1, 2], [0, math.log(1000), 0], tok, 1000)[0], 0)
        self.assertEqual(boxed_certainty([0, 1, 2], [0, 0, 0], tok, 1000)[0], 1)
        self.assertAlmostEqual(probability(.95), .5)
        self.assertEqual(probability(.9), 0)
        with self.assertRaises(ValueError):
            probability(float('nan'))

    def test_nested_braces_and_crossing_tokens(self):
        tok = Tokenizer(['{', '\\frac', '{', '1', '}', '{', '2', '}', '}'])
        self.assertEqual(boxed_certainty(list(range(9)), [0]*9, tok, 100)[1], 'complete')
        self.assertEqual(boxed_certainty(list(range(5)), [0]*5, tok, 100)[0], 0)
        tok = Tokenizer(['{12}'])
        self.assertEqual(boxed_certainty([0], [0], tok, 100)[1], 'no_interior_tokens')

    def test_no_stale_probability_after_boundary_or_end(self):
        s = State('question', Config(interval=2))
        s.accept(1, 'work', {9})
        self.assertTrue(s.accept(9, '\n\n', {9}))
        s.p = 1
        self.assertTrue(s.should_mask())
        s.accept(2, ' ', {9})
        self.assertTrue(s.should_mask())
        s.accept(3, 'But', {9})
        self.assertFalse(s.should_mask())
        s.accept(9, '\n\n', {9})
        self.assertEqual(s.p, 0)
        s.p = 1
        s.accept(151649, '</think>', {9})
        s.accept(151648, '<think>', {9})
        self.assertFalse(s.thinking)
        self.assertFalse(s.should_mask())

    def test_mixed_boundary_content_is_not_masked(self):
        s = State('q', Config(interval=1))
        self.assertFalse(s.accept(9, '\n\nBut', {9}))
        self.assertFalse(s.opening)

    def test_probe_prompt_and_outputs_debit_one_budget(self):
        s = State('q', Config(interval=1, probe_tokens=4, max_tokens=12))
        s.accept(9, '\n\n', {9})
        self.assertTrue(s.reserve_probe(3))
        s.complete_probe(.99, 4)
        self.assertEqual(s.remaining, 4)
        self.assertFalse(s.reserve_probe(3))
        self.assertEqual(s.p, 0)
        self.assertEqual(s.probes, 1)

    def test_rng_independent_of_request_batch_order(self):
        one = {(q, p): draw(q, p, 42) for q in ('a', 'b') for p in range(4)}
        two = {(q, p): draw(q, p, 42) for p in reversed(range(4)) for q in ('b', 'a')}
        self.assertEqual(one, two)
        self.assertNotEqual(one['a', 1], one['b', 1])


class LifecycleTests(unittest.TestCase):
    def test_execution_refuses_missing_batch_authorization(self):
        from run import validate
        with self.assertRaisesRegex(ValueError, 'authorization'):
            validate(dict(phase='engineering', coordination=dict(
                data_reconciled=True, gpu_authorized=False)))

    def test_park_retains_exact_objects_and_restores_on_error(self):
        main = [object(), object()]
        scheduler = SimpleNamespace(running=main, waiting=[], skipped_waiting=[])
        child = object()
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with parked(scheduler) as held:
                self.assertIs(held, main)
                self.assertEqual(scheduler.running, [])
                scheduler.running.append(child)
                raise RuntimeError('failure')
        self.assertEqual(scheduler.running, main+[child])

    def test_waiting_primary_is_not_silently_advanced(self):
        scheduler = SimpleNamespace(running=[], waiting=[object()], skipped_waiting=[])
        with self.assertRaises(RuntimeError):
            with parked(scheduler):
                pass

    def test_off_passes_original_logits_object_and_no_mask(self):
        logits = np.zeros((1, 151936), dtype=np.float32)
        original = logits.copy()
        called = []
        def sampler(x, batch, **kwargs):
            called.append(x)
            return 'sampled'
        owner = SimpleNamespace(torch=None, probes={}, expected_logits={}, states={},
                                suppress=False, callback_host_seconds=0)
        batch = SimpleNamespace(num_draft_tokens=0, num_reqs=1, req_ids=['a'],
                                num_computed_tokens_np=[2], num_scheduled_tokens=[1],
                                prefill_len_np=[3])
        self.assertEqual(SamplerAdapter(owner, sampler)(logits, batch), 'sampled')
        self.assertIs(called[0], logits)
        np.testing.assert_array_equal(logits, original)

    def test_mask_is_request_local_and_excludes_answer_phase(self):
        a, b = State('a', Config()), State('b', Config())
        a.opening = b.opening = True
        a.p = b.p = 1
        b.thinking = False
        owner = SimpleNamespace(torch=None, probes={}, expected_logits={},
                                states={'a': a, 'b': b}, suppress=True,
                                callback_host_seconds=0)
        batch = SimpleNamespace(num_draft_tokens=0, num_reqs=2, req_ids=['a', 'b'],
                                num_computed_tokens_np=[4, 4], num_scheduled_tokens=[1, 1],
                                prefill_len_np=[3, 3])
        logits = np.zeros((2, 151936), dtype=np.float32)
        SamplerAdapter(owner, lambda x, b: x)(logits, batch)
        self.assertEqual(np.isneginf(logits[0]).sum(), len(TRIGGERS))
        self.assertTrue((logits[1] == 0).all())
        self.assertEqual(a.mask_positions, [0])
        self.assertEqual(b.mask_count, 0)

    def test_probe_admission_replays_history_and_disables_new_R(self):
        class Tensor(np.ndarray):
            def zero_(self):
                self[:] = 0
            def copy_(self, source):
                self[:] = source
        history = np.ones((2, 16)).view(Tensor)
        state = SimpleNamespace(_dynamic_indices={'probe': 1}, _history=history,
                                _prompt_lengths={}, _history_lengths={},
                                _in_think=[True, True], _coefs=[1., 1.])
        owner = Backend.__new__(Backend)
        owner.runner = SimpleNamespace(steer_vector_state=state)
        owner.original_add = lambda x: None
        owner.probes = {'probe': {'snapshot': {'history': np.array([0., 0., -.4, .6]),
                                               'prompt_length': 2}}}
        request = SimpleNamespace(req_id='probe', prompt_token_ids=list(range(7)))
        Backend.add_requests(owner, SimpleNamespace(scheduled_new_reqs=[request]))
        np.testing.assert_array_equal(history[0], np.ones(16))
        np.testing.assert_array_equal(history[1, :4], [0., 0., -.4, .6])
        self.assertTrue((history[1, 4:] == 0).all())
        self.assertEqual(state._prompt_lengths['probe'], 2)
        self.assertFalse(state._in_think[1])
        self.assertEqual(state._coefs[1], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
