# SPDX-License-Identifier: Apache-2.0
"""Dynamic ReBalance direction steering."""

import torch

from vllm.forward_context import get_forward_context

from .direct import DirectAlgorithm
from .factory import register_algorithm
from vllm.steer_vectors.graph_kernels import feedback_delta, radial_delta


@register_algorithm("rebalance")
class ReBalanceAlgorithm(DirectAlgorithm):
    """Add a direction scaled by the request's online ReBalance state."""

    def __init__(self, *args, normalize: bool = False, **kwargs):
        if normalize:
            raise ValueError("rebalance does not support normalize=True")
        super().__init__(*args, normalize=False, **kwargs)

    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        ctx = get_forward_context()
        scales = ctx.steer_token_scales
        if scales is None:
            raise RuntimeError("rebalance steering requires per-token scales")

        selected = hidden_states.index_select(0, positions_tensor)
        selected_scales = scales.index_select(0, positions_tensor).to(selected.dtype)
        transformed = selected + selected_scales.unsqueeze(1) * params
        hidden_states.index_copy_(0, positions_tensor, transformed)
        return hidden_states


@register_algorithm("rebalance_feedback")
class ReBalanceFeedbackAlgorithm(ReBalanceAlgorithm):
    """ReBalance state machine with a memoryless negative-displacement limit."""

    graph_family = "feedback"

    @staticmethod
    def graph_lower(payload, scale):
        direction = payload["direction"] * scale
        response = (direction.float() * payload["readout"]).sum()
        if scale <= 0 or not torch.isfinite(response) or response <= 0:
            raise ValueError("feedback needs positive finite static response")
        return {"V": direction, "W": payload["readout"],
                "C": payload["center"], "E": payload["enabled"]}

    @classmethod
    def load_from_path(cls, *args, **kwargs):
        raise ValueError("rebalance_feedback requires FeedbackDirection data")

    def set_payload(self, payload, scale_factor=1.0):
        self._payload = self.graph_lower(payload, scale_factor)

    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        scales = get_forward_context().steer_token_scales
        if scales is None:
            raise RuntimeError("feedback requires ReBalance per-token scales")
        selected = hidden_states.index_select(0, positions_tensor)
        x = selected if residual is None else (
            selected + residual.index_select(0, positions_tensor)
        )
        coefficients = scales.index_select(0, positions_tensor).to(selected.dtype)
        delta = feedback_delta(x, params["V"], params["W"],
                               params["C"], params["E"], coefficients[:, None])
        hidden_states.index_copy_(0, positions_tensor, selected + delta)
        return hidden_states


@register_algorithm("seal")
class SealAlgorithm(ReBalanceAlgorithm):
    """Constant additive direction gated by per-request reasoning markers."""


@register_algorithm("rebalance_radial")
class ReBalanceRadialAlgorithm(ReBalanceAlgorithm):
    """Preserve additive orientation and restore the complete-state norm."""

    graph_family = "radial"
    radial_enabled = True

    @classmethod
    def graph_lower(cls, payload, scale):
        return {"V": payload * scale,
                "E": torch.tensor([float(cls.radial_enabled)],
                                  device=payload.device, dtype=torch.float32)}

    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        scales = get_forward_context().steer_token_scales
        if scales is None:
            raise RuntimeError("radial steering requires ReBalance scales")
        selected = hidden_states.index_select(0, positions_tensor)
        x = selected.float() if residual is None else (
            selected.float() + residual.index_select(0, positions_tensor).float()
        )
        coefficients = scales.index_select(0, positions_tensor).to(selected.dtype)
        enabled = torch.tensor(float(self.radial_enabled), device=selected.device)
        delta = radial_delta(x, params, enabled, coefficients[:, None])
        hidden_states.index_copy_(0, positions_tensor, selected + delta)
        return hidden_states


@register_algorithm("rebalance_radial_disabled")
class ReBalanceRadialDisabledAlgorithm(ReBalanceRadialAlgorithm):
    """Same radial graph with correction disabled, for a matched control."""

    radial_enabled = False
