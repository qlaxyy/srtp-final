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
from backend import Backend, SamplerAdapter, parked, enable_cumulative


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
    def test_real_output_processor_reproduces_and_repairs_missing_checkpoint(self):
        # Execute the pinned production publication method without importing
        # CUDA dependencies. Output construction is stubbed; its gating is real.
        import ast
        from pathlib import Path
        from run import ROOT
        path = ROOT/'sources/EasySteer/vllm-steer/vllm/v1/engine/output_processor.py'
        tree = ast.parse(path.read_text(encoding='utf8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'RequestState')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'make_request_output')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0), method],
                            type_ignores=[])
        kinds = SimpleNamespace(FINAL_ONLY=0, CUMULATIVE=1, DELTA=2)
        scope = {'RequestOutputKind': kinds}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        state = SimpleNamespace(output_kind=kinds.FINAL_ONLY, stream_interval=4,
                                parent_req=None, external_req_id='external')
        state._new_completion_output = lambda *args: args[0]
        state._new_request_output = lambda rid, outputs, finished, *args, **kw: outputs
        publish = scope['make_request_output']
        args = dict(new_token_ids=[9], pooling_output=None, finish_reason=None,
                    stop_reason=None, kv_transfer_params=None, ec_transfer_params=None)
        self.assertIsNone(publish(state, **args))
        request = SimpleNamespace(num_output_tokens=0,
                                  sampling_params=SimpleNamespace(output_kind=0))
        engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=SimpleNamespace(
            scheduler=SimpleNamespace(requests={'a': request}))),
            output_processor=SimpleNamespace(request_states={'a': state}))
        enable_cumulative(SimpleNamespace(llm_engine=engine), ['a'], kinds.CUMULATIVE)
        self.assertEqual(publish(state, **args), [[9]])

    def test_probe_completes_box_aborts_child_and_resumes_only_primary(self):
        import time
        import sys
        owner = Backend.__new__(Backend)
        scheduler = SimpleNamespace(running=['main'], waiting=[], skipped_waiting=[],
                                    requests={})
        output_states = {}
        slots = {'main': 0}
        events, emitted = [], []
        owner.probes = {}
        owner.deadline = time.monotonic() + 5
        owner.tokenizer = Tokenizer(['{', '12', '}', 'unwanted'])
        owner.runner = SimpleNamespace(req_states=SimpleNamespace(req_id_to_index=slots),
                                       _remove_request=lambda rid: slots.pop(rid))
        core = SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler))
        engine = SimpleNamespace(engine_core=core,
                                 output_processor=SimpleNamespace(request_states=output_states))
        def enqueue(prompts, sampling_params, **kw):
            self.assertEqual(prompts, [{'prompt_token_ids': [50, 51, 60]}])
            # Match vLLM's offline override and copied output state.
            sampling_params.output_kind = 'FINAL_ONLY'
            scheduler.requests['child'] = SimpleNamespace(
                num_output_tokens=0, sampling_params=sampling_params)
            output_states['child'] = SimpleNamespace(output_kind='FINAL_ONLY',
                stream_interval=4, external_req_id='external_child')
            scheduler.running.append('child')
            slots['child'] = 1
            return ['child']
        def abort(ids, internal):
            self.assertTrue(internal)
            self.assertEqual(ids, ['child'])
            scheduler.running.remove('child')
            scheduler.requests.pop('child')
            output_states.pop('child')
            events.append('aborted_complete_box')
        def step():
            self.assertNotIn('main', scheduler.running)
            if not scheduler.running:
                return []
            self.assertEqual(output_states['child'].output_kind, 'CUMULATIVE')
            self.assertEqual(output_states['child'].stream_interval, 1)
            emitted.append(len(emitted))
            probe = owner.probes['child']
            probe['entropy'].append(0.)
            probe['vocab_size'] = 1000
            scheduler.requests['child'].num_output_tokens += 1
            return [SimpleNamespace(request_id='external_child', finished=False,
                    outputs=[SimpleNamespace(token_ids=emitted.copy(), finish_reason=None)])]
        engine.step, engine.abort_request = step, abort
        owner.llm = SimpleNamespace(llm_engine=engine, enqueue=enqueue)
        module = SimpleNamespace(RequestOutputKind=SimpleNamespace(CUMULATIVE='CUMULATIVE'))
        with patch.dict(sys.modules, {'vllm.sampling_params': module}):
            with parked(scheduler):
                results = owner.drain_probe_requests(
                    [dict(prefix=[50, 51], snapshot=None, rid='main')],
                    SimpleNamespace(), None, [60])
        self.assertEqual(scheduler.running, ['main'])
        self.assertEqual(slots, {'main': 0})
        self.assertEqual(owner.probes, {})
        self.assertEqual(events, ['aborted_complete_box'])
        self.assertEqual(results[0]['token_ids'], [0, 1, 2])
        certainty, status, _ = boxed_certainty(results[0]['token_ids'],
                            results[0]['entropy'], owner.tokenizer, 1000)
        main = State('main', Config(interval=1))
        self.assertTrue(main.accept(9, '\n\n', {9}))
        self.assertTrue(main.reserve_probe(1))
        main.complete_probe(certainty, 3)
        self.assertEqual(status, 'complete')
        self.assertTrue(main.should_mask())
        main.accept(10, 'answer', {9})
        self.assertFalse(main.should_mask())

    def test_engineering_gate_rejects_missing_probe_mask_or_changed_main(self):
        from copy import deepcopy
        from run import engineering_checks
        row = dict(problem_sha256='q', token_ids=[1, 2], R_history_sha256='history',
                   policy=dict(mask_count=1))
        result = dict(records=[row], probe_output_tokens=3,
                      primary_preservation_checks=[dict(before={'a': 'same'},
                                                         after={'a': 'same'}, passed=True)],
                      replay_checks=[dict(passed=True)])
        good = {a: deepcopy(result) for a in ('R', 'Roff', 'Rshadow', 'C', 'RC')}
        self.assertTrue(all(engineering_checks(good).values()))
        for arm in ('Rshadow', 'C', 'RC'):
            bad = deepcopy(good)
            bad[arm]['probe_output_tokens'] = 0
            self.assertFalse(all(engineering_checks(bad).values()))
        for arm in ('C', 'RC'):
            bad = deepcopy(good)
            bad[arm]['records'][0]['policy']['mask_count'] = 0
            self.assertFalse(all(engineering_checks(bad).values()))
        bad = deepcopy(good)
        bad['Rshadow']['records'][0]['token_ids'] = [1, 3]
        self.assertFalse(engineering_checks(bad)['Rshadow_equivalent'])
        bad = deepcopy(good)
        bad['Rshadow']['replay_checks'] = []
        self.assertFalse(engineering_checks(bad)['replay_passed'])
        bad = deepcopy(good)
        bad['RC']['primary_preservation_checks'][0]['after'] = {'a': 'changed'}
        self.assertFalse(engineering_checks(bad)['RC_preserved'])

    def test_offline_final_only_override_is_repaired_in_copied_output_state(self):
        request = SimpleNamespace(num_output_tokens=0,
                                  sampling_params=SimpleNamespace(output_kind='FINAL_ONLY'))
        state = SimpleNamespace(output_kind='FINAL_ONLY', stream_interval=4)
        engine = SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(
                scheduler=SimpleNamespace(requests={'a': request}))),
            output_processor=SimpleNamespace(request_states={'a': state}))
        llm = SimpleNamespace(llm_engine=engine)
        enable_cumulative(llm, ['a'], 'CUMULATIVE')
        self.assertEqual(state.output_kind, 'CUMULATIVE')
        self.assertEqual(state.stream_interval, 1)
        self.assertEqual(request.sampling_params.output_kind, 'CUMULATIVE')
        request.num_output_tokens = 1
        with self.assertRaises(RuntimeError):
            enable_cumulative(llm, ['a'], 'CUMULATIVE')

    def test_archive_deployment_does_not_require_git_metadata(self):
        import tempfile
        from pathlib import Path
        from run import deployment_record
        with tempfile.TemporaryDirectory() as folder:
            result = deployment_record(Path(folder))
            self.assertFalse(result['git_metadata_available'])
            self.assertIsNone(result['git_commit'])

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
