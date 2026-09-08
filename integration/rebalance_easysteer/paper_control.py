"""ReBalance paper Appendix B.3 control surface, with explicit constants.

The PDF page 32 contains abs(c-c_high) inside tanh. Constants absent from
the paper have no defaults here: callers must provide and record them.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PaperControl:
    confidence_low: float
    confidence_high: float
    variance_low: float
    variance_high: float
    moderate_amplitude: float
    overthinking_amplitude: float
    underthinking_amplitude: float
    confidence_gate_width: float
    variance_gate_width: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in self.__dict__.values()):
            raise ValueError("Paper control constants must be finite")
        if not 0 <= self.confidence_low < self.confidence_high <= 1:
            raise ValueError("Invalid confidence thresholds")
        if not 0 <= self.variance_low < self.variance_high <= .25:
            raise ValueError("Invalid two-step variance thresholds")
        if min(self.moderate_amplitude, self.overthinking_amplitude,
               self.underthinking_amplitude) < 0:
            raise ValueError("Amplitudes must be nonnegative")
        if min(self.confidence_gate_width, self.variance_gate_width) <= 0:
            raise ValueError("Gate widths must be positive")


def coefficient(confidence, variance, params: PaperControl):
    """Evaluate the smooth Appendix B.3 surface; no author-code clamp/offset."""
    over = torch.sigmoid((params.confidence_low-confidence) /
                         params.confidence_gate_width)
    over *= torch.sigmoid((variance-params.variance_high) /
                          params.variance_gate_width)
    under = torch.sigmoid((confidence-params.confidence_high) /
                          params.confidence_gate_width)
    under *= torch.sigmoid((params.variance_low-variance) /
                           params.variance_gate_width)
    amplitude = params.moderate_amplitude
    amplitude = amplitude + (params.overthinking_amplitude -
                             params.moderate_amplitude) * over
    amplitude = amplitude + (params.underthinking_amplitude -
                             params.moderate_amplitude) * under
    delta = confidence-params.confidence_high
    return torch.sign(delta) * amplitude * torch.tanh(delta.abs())


def geometric_confidence(log_probability_sum, token_count):
    """A completed nonempty step; callers gate zero-count steps separately."""
    return torch.exp(log_probability_sum / token_count.clamp_min(1))


class PaperStepState:
    """Reference batched online state, in stable request-slot order.

    Returned scales belong to the sampled tokens when subsequently fed back
    into the model. Only the first content token after a boundary is steered.
    The first step is unsteered because no completed-step confidence exists.
    This reference is not yet connected to the vLLM request scheduler.
    """

    def __init__(self, count, device="cpu"):
        self.log_sum = torch.zeros(count, device=device)
        self.count = torch.zeros(count, dtype=torch.long, device=device)
        self.previous = torch.full((count,), torch.nan, device=device)
        self.strength = torch.zeros(count, device=device)
        self.in_think = torch.ones(count, dtype=torch.bool, device=device)
        self.pending_first = torch.ones_like(self.in_think)

    def observe(self, tokens, max_probabilities, boundaries, end_id, params):
        boundary = (tokens[:, None] == boundaries).any(dim=-1)
        self.in_think &= tokens != end_id
        content = self.in_think & ~boundary
        inject = content & self.pending_first
        scales = torch.where(inject, self.strength, 0.)
        self.pending_first = torch.where(boundary, True, self.pending_first)
        self.pending_first &= ~content & self.in_think
        safe = max_probabilities.clamp_min(torch.finfo(torch.float32).tiny)
        self.log_sum += torch.where(content, safe.log(), 0.)
        self.count += content
        ready = boundary & self.in_think & (self.count > 0)
        confidence = geometric_confidence(self.log_sum, self.count)
        variance = torch.where(torch.isfinite(self.previous),
            (confidence-self.previous).square()/4., 0.)
        updated = coefficient(confidence, variance, params)
        self.strength = torch.where(ready, updated, self.strength)
        self.previous = torch.where(ready, confidence, self.previous)
        reset = ready | ~self.in_think
        self.log_sum = torch.where(reset, 0., self.log_sum)
        self.count = torch.where(reset, 0, self.count)
        return scales
