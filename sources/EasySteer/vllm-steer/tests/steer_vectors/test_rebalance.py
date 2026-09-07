from types import SimpleNamespace

import torch

from vllm.steer_vectors.rebalance import (
    ReBalanceParams,
    compute_rebalance_coefficient,
)
from vllm.v1.worker.gpu.steer_vector_utils import SteerVectorState


def test_rebalance_controller_hits_published_anchor_values():
    params = ReBalanceParams(
        boundary_token_ids=(99,),
        think_start_token_id=10,
        think_end_token_id=11,
        q25c=0.662293,
        q75c=0.94805,
        low_val_1=-1.02,
        q25v=0.000560,
        q75v=0.011597,
        low_val_2=-1.91,
        high_val_2=0.1,
    )
    confidence = torch.tensor([params.q25c, 1.0])
    variance = torch.tensor([params.q75v, params.q25v])

    actual = compute_rebalance_coefficient(confidence, variance, params)

    torch.testing.assert_close(
        actual,
        torch.tensor([params.low_val_2, params.high_val_2]),
        atol=1e-5,
        rtol=1e-5,
    )


class _Manager:
    def acquire_config(self, req_id, request):
        return 0

    def release_config(self, req_id):
        pass


def _request():
    return SimpleNamespace(
        algorithm="rebalance",
        rebalance_boundary_token_ids=[99],
        rebalance_think_start_token_id=10,
        rebalance_think_end_token_id=11,
        rebalance_initial_coef=-1.0,
        rebalance_q25c=0.65,
        rebalance_q75c=0.90,
        rebalance_low_val_1=-1.0,
        rebalance_q25v=0.0005,
        rebalance_q75v=0.01,
        rebalance_low_val_2=-2.0,
        rebalance_high_val_2=0.1,
    )


def test_rebalance_state_is_request_local_and_uses_arithmetic_mean():
    state = SteerVectorState(max_num_reqs=2, device=torch.device("cpu"))
    manager = _Manager()
    for index, req_id in enumerate(("a", "b")):
        state.add_request(
            req_id,
            _request(),
            manager,
            req_index=index,
            prompt_token_ids=[10],
        )
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_draft_tokens=0,
    )

    state.observe_sample(
        batch,
        torch.tensor([[1], [2]], dtype=torch.int32),
        torch.tensor([0.8, 0.4]),
    )
    state.observe_sample(
        batch,
        torch.tensor([[3], [4]], dtype=torch.int32),
        torch.tensor([0.6, 0.2]),
    )
    state.observe_sample(
        batch,
        torch.tensor([[99], [99]], dtype=torch.int32),
        torch.tensor([0.1, 0.9]),
    )

    params = ReBalanceParams.from_request(_request())
    expected = compute_rebalance_coefficient(
        torch.tensor([0.7, 0.3]),
        torch.zeros(2),
        params,
    )
    torch.testing.assert_close(state.batch_scales(batch), expected)
