"""Objective primitives only; no model loading, controller, or GPU runner.

This is a unidirectional preference-inspired adaptation, not official BiPO.
Each input sequence score must include the full response and exclude prompt/pad.
"""
import torch
import torch.nn.functional as F


def causal_response_logp(logits, input_ids, response_mask):
    """Mask is on target token positions, NOT predictor positions."""
    if logits.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError('incompatible sequence dimensions')
    if response_mask[:, 0].any():
        raise ValueError('first token has no within-sequence causal predictor')
    mask = response_mask[:, 1:].bool()
    count = mask.sum(-1)
    if (count == 0).any():
        raise ValueError('empty response')
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    selected = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    score = selected.masked_fill(~mask, 0).sum(-1)
    return score, score / count, count


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    """One scalar per QUESTION; reference tensors never receive gradients."""
    if beta <= 0 or not (chosen.shape == rejected.shape == reference_chosen.shape == reference_rejected.shape):
        raise ValueError('invalid beta or pair shapes')
    margin = (chosen - reference_chosen.detach()) - (rejected - reference_rejected.detach())
    return -F.logsigmoid(beta * margin), margin


def negative_branch_residual(alpha, delta):
    """Additional vector only. Original alpha*d0 remains applied separately."""
    return torch.clamp(-alpha, min=0).unsqueeze(-1) * delta


def cpu_check():
    torch.manual_seed(42)
    ids = torch.tensor([[1, 2, 3, 0, 1], [2, 1, 0, 3, 2]])
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=torch.bool)
    base = torch.randn(2, 5, 4)
    shift = torch.randn_like(base)
    delta = torch.tensor(0.0, requires_grad=True)
    ref, _, counts = causal_response_logp(base, ids, mask)
    scores, _, _ = causal_response_logp(base + delta*shift, ids, mask)
    loss, margin = preference_loss(scores[:1], scores[1:], ref[:1], ref[1:])
    assert torch.equal(margin, torch.zeros_like(margin))
    loss.sum().backward()
    def value(z):
        s, _, _ = causal_response_logp(base + z*shift, ids, mask)
        return preference_loss(s[:1], s[1:], ref[:1], ref[1:])[0].sum().item()
    numeric = (value(0.01) - value(-0.01))/0.02
    assert abs(numeric-delta.grad.item()) < 2e-4
    altered = base.clone()
    # Changes to unused predictor positions must not change scored responses.
    predictor_mask = torch.zeros_like(mask)
    predictor_mask[:, :-1] = mask[:, 1:]
    altered[~predictor_mask] += 100*shift[~predictor_mask]
    assert torch.equal(causal_response_logp(altered, ids, mask)[0], ref)
    alpha = torch.tensor([-1.0, -0.5, 0.0, 0.1])
    zero = negative_branch_residual(alpha, torch.zeros(3))
    assert torch.equal(zero, torch.zeros(4, 3))
    active = negative_branch_residual(alpha, torch.ones(3))
    assert torch.equal(active[2:], torch.zeros(2, 3))
    return dict(status='CPU tensor checks passed; NOT model or vLLM equivalence',
        counts=counts.tolist(), zero_margin_loss=loss.item(),
        analytic_gradient=delta.grad.item(), finite_difference_gradient=numeric,
        checks=['causal target alignment', 'prompt/padding exclusion',
                'zero-update log(2) loss', 'nonzero usable gradient at zero initialization',
                'zero residual identity', 'positive branch unchanged'])


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_check(), indent=2))
