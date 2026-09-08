from types import SimpleNamespace

import torch

from vllm.steer_vectors.rebalance import (
    ReBalanceParams,
    compute_paper_coefficient,
    compute_rebalance_coefficient,
)
from vllm.v1.worker.gpu.steer_vector_utils import SteerVectorState


def test_auto_calibration_rejects_infeasible_fit_and_hits_all_three_anchors():
    """Catch the silent k-floor failure missed by final-surface endpoint checks."""
    from vllm.steer_vectors.rebalance import (
        _baseline, _curve_constants, validate_curve_targets,
    )
    cl, ch, low = .8669240897121249, .9989939076206545, -.3394509798752199
    try:
        validate_curve_targets(cl, ch, low, .01)
    except ValueError as error:
        assert "Infeasible curve" in str(error)
    else:
        raise AssertionError("Old self-calibration must not silently pass")
    tau = min(.01, .5 * -low * (1-ch)/(ch-cl))
    validate_curve_targets(cl, ch, low, tau)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dtype in (torch.float32, torch.float64):
            mid, k, a, b, _, at_one = _curve_constants(
                cl, ch, low, torch.device(device), dtype, tau
            )
            assert 1e-6 < k < 1e6
            c = torch.tensor([cl, ch, 1.], device=device, dtype=dtype)
            torch.testing.assert_close(
                _baseline(c, mid, k, a, b),
                torch.tensor([low, 0., tau], device=device, dtype=dtype),
                atol=3e-7, rtol=1e-5,
            )


def test_paper_next_content_reordering_and_slot_reuse():
    """Geometric confidence must follow request slots and steer only one token."""
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        req = _request()
        req.rebalance_initial_coef = 0.0
        req.rebalance_paper_parameters = [2., 2., 2., .02, .002]
        state = SteerVectorState(3, torch.device(device))
        manager = _Manager()
        for i, name in enumerate(("a", "b")):
            state.add_request(name, req, manager, req_index=i, prompt_token_ids=[10])
        batch = SimpleNamespace(
            req_ids=["b", "plain", "a"], num_reqs=3, num_draft_tokens=0,
            idx_mapping=torch.tensor([1, 2, 0], device=device),
        )

        def observe(tokens, probs):
            state.observe_sample(
                batch, torch.tensor(tokens, device=device)[:, None],
                torch.tensor(probs, device=device),
            )
            return state.batch_scales(batch)

        observe([1, 1, 1], [.9, .5, .4])
        observe([2, 2, 2], [.1, .5, .4])
        scales = observe([99, 2, 99], [.99, .5, .01])
        torch.testing.assert_close(scales, torch.tensor([0., 1., 0.], device=device))
        torch.testing.assert_close(
            state._prev_step_mean[:2], torch.tensor([.4, .3], device=device)
        )
        observe([99, 2, 99], [.9, .5, .9])
        scales = observe([3, 2, 3], [.5, .5, .5])
        expected = 2 * torch.tanh(torch.tensor([.3, .4], device=device) - .9)
        torch.testing.assert_close(scales[[0, 2]], expected)
        scales = observe([4, 2, 4], [.5, .5, .5])
        torch.testing.assert_close(scales[[0, 2]], torch.zeros(2, device=device))
        observe([99, 2, 99], [.9, .5, .9])
        scales = observe([11, 2, 3], [.9, .5, .5])
        assert scales[0] == 0 and scales[2] < 0
        state.remove_request("b", manager)
        state.add_request("new", req, manager, req_index=1, prompt_token_ids=[10])
        batch.req_ids[0] = "new"
        assert state.batch_scales(batch)[0] == 0
        assert state._step_tok_count[1] == 0


def test_paper_surface_sign_and_explicit_parameter_wire():
    """Paper opt-in survives API/wire conversion without changing legacy defaults."""
    import msgspec
    from vllm.steer_vectors.api import ApplySpec, SteeringSpec, VectorSpec, to_engine_request
    from vllm.steer_vectors.request import SteerVectorRequest

    spec = SteeringSpec(vectors=[VectorSpec(
        name="paper", source="/unused.pt", layers=[22], algorithm="rebalance",
        apply=ApplySpec(generation="all"), params=dict(
            boundary_token_ids=[99], think_start_token_id=10,
            think_end_token_id=11, initial_coef=0.,
            paper_parameters=[2., 2., 2., .02, .002],
            curve_tau=.004,
        ),
    )])
    req = to_engine_request(spec, name="paper", int_id=1)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=SteerVectorRequest)
    params = ReBalanceParams.from_request(decoded)
    assert params.paper_parameters == (2., 2., 2., .02, .002)
    assert params.curve_tau == .004
    assert ReBalanceParams.from_request(_request()).paper_parameters is None
    c = torch.tensor([.1, .5, .9, 1.])
    torch.testing.assert_close(
        compute_paper_coefficient(c, torch.zeros_like(c), params),
        2 * torch.tanh(c - .9),
    )


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
    confidence = torch.tensor([params.q25c, 1.0, 0.7, 0.8, 0.9])
    variance = torch.tensor([params.q75v, params.q25v, 0.001, 0.001, 0.001])

    actual = compute_rebalance_coefficient(confidence, variance, params)

    torch.testing.assert_close(
        actual,
        # Interior values from the executable author build_F/default high_val=0.
        torch.tensor([-1.91, 0.1, -0.99339795, -0.55161738, -0.04057819]),
        atol=1e-5,
        rtol=1e-5,
    )


class _Manager:
    def acquire_config(self, req_id, request):
        return 0

    def release_config(self, req_id):
        pass


def test_rebalance_boundary_matches_isin():
    # Includes repeated sampled IDs, absent IDs, and an empty boundary set.
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for size in (1, 20, 256):
            tokens = torch.arange(size, device=device) % 37 - 1
            for count in (0, 1, 565):
                boundaries = torch.arange(count, device=device) * 2
                torch.testing.assert_close(
                    SteerVectorState._is_boundary(tokens, boundaries),
                    torch.isin(tokens, boundaries),
                )


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


def test_rebalance_reordering_think_end_and_slot_reuse():
    state = SteerVectorState(max_num_reqs=3, device=torch.device("cpu"))
    manager = _Manager()
    for index, req_id in enumerate(("a", "b")):
        state.add_request(req_id, _request(), manager, req_index=index,
                          prompt_token_ids=[10])
    batch = SimpleNamespace(
        req_ids=["b", "a", "plain"],
        idx_mapping=torch.tensor([1, 0, 2], dtype=torch.int32),
        num_reqs=3, num_draft_tokens=0,
    )
    state.observe_sample(batch, torch.tensor([[1], [1], [1]]),
                         torch.tensor([0.8, 0.3, 0.9]))
    state.observe_sample(batch, torch.tensor([[99], [99], [99]]),
                         torch.tensor([0.1, 0.9, 0.2]))
    expected = compute_rebalance_coefficient(
        torch.tensor([0.8, 0.3]), torch.zeros(2),
        ReBalanceParams.from_request(_request()),
    )
    torch.testing.assert_close(state.batch_scales(batch),
                               torch.cat([expected, torch.ones(1)]))
    # Consecutive boundaries must not replace the previous nonempty step.
    state.observe_sample(batch, torch.tensor([[99], [99], [99]]),
                         torch.tensor([0.9, 0.1, 0.2]))
    torch.testing.assert_close(state.batch_scales(batch)[:2], expected)
    state.observe_sample(batch, torch.tensor([[11], [1], [1]]),
                         torch.tensor([0.9, 0.1, 0.2]))
    assert state.batch_scales(batch)[0] == 0
    state.remove_request("b", manager)
    state.add_request("new", _request(), manager, req_index=1,
                      prompt_token_ids=[10])
    batch.req_ids[0] = "new"
    assert state.batch_scales(batch)[0] == -1
    assert state._step_tok_count[1] == 0
    assert torch.isnan(state._prev_step_mean[1])


def test_rebalance_cached_positions_follow_changed_batch_membership():
    state = SteerVectorState(max_num_reqs=3, device=torch.device("cpu"))
    manager = _Manager()
    for index, req_id in enumerate(("a", "b")):
        state.add_request(req_id, _request(), manager, req_index=index,
                          prompt_token_ids=[10])
    state._coefs[:2] = torch.tensor([-0.2, -0.8])
    batch = SimpleNamespace(
        req_ids=["a", "plain", "b"], num_reqs=3,
        idx_mapping=torch.tensor([0, 2, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(state.batch_scales(batch),
                               torch.tensor([-0.2, 1.0, -0.8]))
    # Identical position pattern, but a different live request-to-state mapping.
    batch.req_ids = ["b", "plain", "a"]
    batch.idx_mapping = torch.tensor([1, 2, 0], dtype=torch.int32)
    torch.testing.assert_close(state.batch_scales(batch),
                               torch.tensor([-0.8, 1.0, -0.2]))
    # The next scheduler batch changes the position pattern itself.
    batch.req_ids = ["plain", "b", "a"]
    batch.idx_mapping = torch.tensor([2, 1, 0], dtype=torch.int32)
    torch.testing.assert_close(state.batch_scales(batch),
                               torch.tensor([1.0, -0.8, -0.2]))
    state.remove_request("a", manager)
    state.remove_request("b", manager)
    assert not state._position_tensors
