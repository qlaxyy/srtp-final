# SPDX-License-Identifier: Apache-2.0
"""ReBalance's published online confidence controller."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ReBalanceParams:
    """Parameters and token markers for one ReBalance request."""

    boundary_token_ids: tuple[int, ...]
    think_start_token_id: int
    think_end_token_id: int
    initial_coef: float = -1.0
    q25c: float = 0.65
    q75c: float = 0.90
    low_val_1: float = -1.0
    q25v: float = 0.0005
    q75v: float = 0.01
    low_val_2: float = -2.0
    high_val_2: float = 0.1

    @classmethod
    def from_request(cls, request) -> "ReBalanceParams":
        """Create parameters from an engine-side steering request."""
        boundary_ids = request.rebalance_boundary_token_ids
        start_id = request.rebalance_think_start_token_id
        end_id = request.rebalance_think_end_token_id
        if not boundary_ids or start_id is None or end_id is None:
            raise ValueError("incomplete ReBalance request parameters")
        return cls(
            boundary_token_ids=tuple(boundary_ids),
            think_start_token_id=start_id,
            think_end_token_id=end_id,
            initial_coef=request.rebalance_initial_coef,
            q25c=request.rebalance_q25c,
            q75c=request.rebalance_q75c,
            low_val_1=request.rebalance_low_val_1,
            q25v=request.rebalance_q25v,
            q75v=request.rebalance_q75v,
            low_val_2=request.rebalance_low_val_2,
            high_val_2=request.rebalance_high_val_2,
        )


def _solve_k_for_tau(
    q25: float,
    q75: float,
    low_val: float,
    high_val: float,
    tau: float,
    iters: int = 80,
) -> float:
    midpoint = 0.5 * (q25 + q75)
    half_width = max(1e-9, 0.5 * (q75 - q25))
    denominator = max(high_val - low_val, 1e-12)
    target = 2.0 * (tau - 0.5 * (low_val + high_val)) / denominator

    def ratio(k: float) -> float:
        numerator = math.tanh(k * max(0.0, 1.0 - midpoint))
        scaled = k * half_width
        denominator = math.tanh(scaled) if scaled > 1e-12 else scaled
        return numerator / denominator if denominator > 0 else math.inf

    low, high = 1e-6, 1.0
    while ratio(high) > target and high < 1e6:
        high *= 2.0
    for _ in range(iters):
        middle = 0.5 * (low + high)
        if ratio(middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def compute_rebalance_coefficient(
    confidence: torch.Tensor,
    variance: torch.Tensor,
    params: ReBalanceParams,
) -> torch.Tensor:
    """Map step confidence and the two-step variance proxy to a coefficient."""
    q25c, q75c = sorted((params.q25c, params.q75c))
    q25v, q75v = sorted((params.q25v, params.q75v))
    high_val_1 = 0.01

    midpoint = 0.5 * (q25c + q75c)
    half_width = max(1e-9, 0.5 * (q75c - q25c))
    k = _solve_k_for_tau(
        q25c,
        q75c,
        params.low_val_1,
        high_val_1,
        high_val_1,
    )
    span = math.tanh(k * half_width)
    intercept = 0.5 * (params.low_val_1 + high_val_1)
    slope = (high_val_1 - params.low_val_1) / (2.0 * max(span, 1e-12))

    def baseline(values: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(
            values, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        return intercept + slope * torch.tanh(k * (values - midpoint))

    iqrc = max(1e-12, q75c - q25c)
    iqrv = max(1e-12, q75v - q25v)

    def sigmoid(values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(values.clamp(-60.0, 60.0))

    low_gate = sigmoid((q25c - confidence) / iqrc * 1200.0)
    high_gate = sigmoid((confidence - q75c) / iqrc * 1200.0)
    high_variance_gate = sigmoid((variance - q75v) / iqrv * 1200.0)
    low_variance_gate = sigmoid((q25v - variance) / iqrv * 1200.0)

    def scalar_sigmoid(value: float) -> float:
        value = max(min(value, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-value))

    low_normalizer = scalar_sigmoid(0.0) * scalar_sigmoid(0.0)
    high_normalizer = scalar_sigmoid(
        (1.0 - q75c) / iqrc * 12.0
    ) * scalar_sigmoid(0.0)
    low_weight = (low_gate * high_variance_gate / low_normalizer).clamp(max=1.0)
    high_weight = (
        high_gate * low_variance_gate / high_normalizer
    ).clamp(max=1.0)

    at_q25 = baseline(confidence.new_tensor(q25c))
    at_one = baseline(confidence.new_tensor(1.0))
    updated = baseline(confidence)
    updated += (params.low_val_2 - at_q25) * low_weight
    updated += (params.high_val_2 - at_one) * high_weight
    return updated.clamp(
        min=min(params.low_val_2, params.low_val_1),
        max=max(high_val_1, params.high_val_2),
    )
