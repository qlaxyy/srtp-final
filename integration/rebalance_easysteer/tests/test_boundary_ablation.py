"""CPU checks for fixed sampling and exact one-boundary state differences."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'integration/rebalance_easysteer/scripts'))
from prepare_boundary_ablation import completed_boundaries, select_cases


class SampleChecks(unittest.TestCase):
    def test_boundary_state_does_not_read_future(self):
        ids, logs = [2,3,99,4,5,99,11], [-.1,-.2,-.8,-.4,-.1,-.3,-.2]
        a = list(completed_boundaries(ids, logs, {99}, 11))
        b = list(completed_boundaries(ids[:3]+[77,11], logs[:3]+[-9.,-8.], {99}, 11))
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0]['boundary_output_offset'], 2)
        self.assertEqual(a[0]['step_tokens'], 2)

    def test_empty_and_ended_steps_are_not_candidates(self):
        rows = list(completed_boundaries([99,99,3,99,11,3,99], [-.1]*7, {99}, 11))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['boundary_output_offset'], 3)

    def test_question_split_and_no_replacement(self):
        candidates = {i:dict(strong_negative=[{'step':i}], positive=[{'step':i}]) for i in range(30)}
        cases, _ = select_cases(candidates, [0,1,2])
        keys = [c['calibration_index'] for c in cases]
        self.assertEqual(len(keys), 20)
        self.assertEqual(len(set(keys)), 20)
        self.assertFalse(set(keys) & {0,1,2})
        self.assertEqual(cases, select_cases(candidates, [0,1,2])[0])


try:
    import torch
    import msgspec
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    from vllm.steer_vectors.api import to_engine_request
    from vllm.steer_vectors.request import SteerVectorRequest
    from vllm.steer_vectors.rebalance import ReBalanceParams, compute_rebalance_coefficient
    from vllm.v1.worker.gpu.steer_vector_utils import SteerVectorState
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False


@unittest.skipUnless(HAVE_RUNTIME, 'Requires existing server torch/vLLM environment; CPU only')
class StateChecks(unittest.TestCase):
    def request(self, apply=None):
        hp = json.loads((ROOT/'integration/rebalance_easysteer/configs/auto_code_v2_1p5b_20260908.json').read_text())['parameters']
        hp.update(boundary_token_ids=[99], think_start_token_id=10, think_end_token_id=11)
        if apply is not None:
            hp.update(prefix_mean=.8, prefix_variance=.01, prefix_apply=apply)
        return to_engine_request(SteeringSpec(vectors=[VectorSpec(source='/not-loaded.pt',
            algorithm='rebalance', layers=[20], normalize=False, params=hp,
            apply=ApplySpec(prompt_positions=[-1], generation_tokens=[99]))]))

    def pair(self):
        state = SteerVectorState(3, torch.device('cpu'), 64)
        manager = SimpleNamespace(acquire_config=lambda *a:0, release_config=lambda *a:None)
        requests = [self.request(True), self.request(False)]
        for i, request in enumerate(requests):
            state.add_request(str(i), request, manager, req_index=i, prompt_token_ids=[10,4,5,99])
        return state, manager, requests

    def test_wire_round_trip_and_validation(self):
        original = self.request(False)
        restored = msgspec.msgpack.decode(msgspec.msgpack.encode(original), type=SteerVectorRequest)
        self.assertFalse(restored.rebalance_prefix_apply)
        self.assertEqual(restored.rebalance_prefix_mean, .8)
        with self.assertRaises(ValueError):
            SteerVectorRequest(steer_vector_name='invalid', steer_vector_int_id=1,
                steer_vector_local_path='/not-loaded.pt', algorithm='rebalance',
                rebalance_prefix_mean=.8)

    def test_only_last_prefix_position_differs(self):
        state, _, requests = self.pair()
        expected = compute_rebalance_coefficient(torch.tensor([.8]), torch.tensor([.01]),
            ReBalanceParams.from_request(requests[0]))[0]
        torch.testing.assert_close(state._coefs[:2], expected.repeat(2))
        torch.testing.assert_close(state._prev_step_mean[:2], torch.tensor([.8,.8]))
        self.assertEqual(torch.nonzero(state._history[0] != state._history[1]).flatten().tolist(), [3])
        self.assertEqual(float(state._history[1,3]), 0.)
        self.assertEqual(int(state._step_tok_count.sum()), 0)

    def test_future_control_and_eviction_preserve_skip(self):
        import numpy as np
        state, manager, requests = self.pair()
        for n, token in enumerate([7,99]):
            batch = SimpleNamespace(num_reqs=2, num_draft_tokens=0, req_ids=['0','1'],
                idx_mapping=torch.tensor([0,1]), seq_lens=torch.tensor([4+n,4+n]),
                num_computed_tokens_np=np.array([3+n,3+n]), num_scheduled_tokens=np.array([1,1]))
            state.observe_sample(batch, torch.tensor([[token],[token]]), torch.tensor([.95,.95]))
        torch.testing.assert_close(state._coefs[0], state._coefs[1])
        torch.testing.assert_close(state._history[0,4:6], state._history[1,4:6])
        history = state._history[1,:6].clone()
        coef = state._coefs[1].clone()
        state.suspend_request('1')
        state.remove_request('1', manager)
        state.add_request('1', requests[1], manager, req_index=2,
            prompt_token_ids=[10,4,5,99], num_generated_tokens=2)
        torch.testing.assert_close(state._history[2,:6], history)
        torch.testing.assert_close(state._coefs[2], coef)
        self.assertEqual(state.replay_counts, dict(suspended=1, restored=1))

    def test_ordinary_request_keeps_original_initial_behavior(self):
        state = SteerVectorState(1, torch.device('cpu'), 64)
        manager = SimpleNamespace(acquire_config=lambda *a:0)
        state.add_request('ordinary', self.request(), manager, req_index=0, prompt_token_ids=[10,5])
        self.assertEqual(float(state._coefs[0]), -1.)
        self.assertEqual(float(state._history.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
