"""Fast CPU checks for the ReBalance vLLM integration."""

from types import SimpleNamespace

import torch

from vllm.steer_vectors.algorithms import get_algorithm
from vllm.steer_vectors.rebalance import (
    ReBalanceParams,
    compute_rebalance_coefficient,
)
from vllm.v1.worker.gpu.steer_vector_utils import SteerVectorState


class Manager:
    def acquire_config(self, req_id, request):
        return 0

    def release_config(self, req_id):
        pass


def request():
    return SimpleNamespace(
        algorithm="rebalance",
        rebalance_boundary_token_ids=[99],
        rebalance_think_start_token_id=10,
        rebalance_think_end_token_id=11,
        rebalance_initial_coef=-1.0,
        rebalance_q25c=0.662293,
        rebalance_q75c=0.94805,
        rebalance_low_val_1=-1.02,
        rebalance_q25v=0.000560,
        rebalance_q75v=0.011597,
        rebalance_low_val_2=-1.91,
        rebalance_high_val_2=0.1,
    )


def main() -> None:
    algorithm = get_algorithm("rebalance")
    assert algorithm.graph_family == "additive"
    params = ReBalanceParams.from_request(request())
    anchors = compute_rebalance_coefficient(
        torch.tensor([params.q25c, 1.0]),
        torch.tensor([params.q75v, params.q25v]),
        params,
    )
    torch.testing.assert_close(
        anchors,
        torch.tensor([params.low_val_2, params.high_val_2]),
        atol=1e-5,
        rtol=1e-5,
    )

    state = SteerVectorState(max_num_reqs=2, device=torch.device("cpu"))
    manager = Manager()
    for index, req_id in enumerate(("a", "b")):
        state.add_request(
            req_id,
            request(),
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
    for tokens, probabilities in (
        ([[1], [2]], [0.8, 0.4]),
        ([[3], [4]], [0.6, 0.2]),
        ([[99], [99]], [0.1, 0.9]),
    ):
        state.observe_sample(
            batch,
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor(probabilities),
        )
    expected = compute_rebalance_coefficient(
        torch.tensor([0.7, 0.3]),
        torch.zeros(2),
        params,
    )
    torch.testing.assert_close(state.batch_scales(batch), expected)
    print("ReBalance controller, arithmetic mean, and request isolation: OK")


if __name__ == "__main__":
    main()
