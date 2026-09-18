"""CPU reference for a proposed, NOT GPU-validated L27 strength replacement.

This is our design, not VCM or a correctness detector. Candidate IDs and masks
must come from the unchanged L27 matcher and negative-ReBalance gate.
"""
import math
import torch


def reflection_margin_penalty(logits, ids, mask, enabled=False):
    if not enabled:
        return logits
    if logits.ndim != 2 or mask.shape != (logits.shape[0], len(ids)):
        raise ValueError('One mask entry per row and candidate column required')
    if len(set(ids)) != len(ids) or any(i < 0 or i >= logits.shape[1] for i in ids):
        raise ValueError('Unique valid candidate IDs required')
    if mask.dtype != torch.bool:
        raise ValueError('Boolean mask required')
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError('NaN or positive infinity is invalid')
    if not len(ids) or not mask.any():
        return logits
    values = logits[:, ids]
    active = mask & torch.isfinite(values)
    reflected = torch.where(active, values, -torch.inf).float().amax(-1)
    alternatives = logits.float().clone()
    alternatives[:, ids] = torch.where(active, -torch.inf, alternatives[:, ids])
    best_other = alternatives.amax(-1)
    valid = torch.isfinite(reflected) & torch.isfinite(best_other)
    gap = torch.where(valid, (reflected-best_other).clamp_min(0), 0.)
    amount = (math.log(2)-gap).clamp(0, math.log(2))
    amount = torch.where(valid, amount, 0.)
    # Preserve the exact native fixed-penalty arithmetic whenever it applies.
    fixed = values-math.log(2)
    adapted = (values.float()-amount[:, None]).to(logits.dtype)
    adjusted = torch.where((gap == 0)[:, None] & valid[:, None], fixed, adapted)
    adjusted = torch.where((amount == 0)[:, None], values, adjusted)
    result = logits.clone()
    result[:, ids] = torch.where(active, adjusted, values)
    return result


def cpu_checks():
    names=[]
    for dtype in (torch.float32,torch.bfloat16):
        x=torch.tensor([[1.,2.,0.],[2.25,2.,0.],[4.,2.,0.],[2.,-torch.inf,0.]],dtype=dtype)
        mask=torch.tensor([[True],[True],[True],[False]])
        assert reflection_margin_penalty(x,[0],mask) is x
        result=reflection_margin_penalty(x,[0],mask,True)
        assert torch.equal(result[0,0],x[0,0]-math.log(2))
        assert x[1,0]-math.log(2) <= result[1,0] <= x[1,0]
        assert torch.equal(result[2],x[2]) and torch.equal(result[3],x[3])
        assert torch.equal(result[:,1:],x[:,1:]) and result.dtype == x.dtype
        assert torch.equal(x,torch.tensor([[1.,2.,0.],[2.25,2.,0.],[4.,2.,0.],[2.,-torch.inf,0.]],dtype=dtype))
        assert reflection_margin_penalty(x,[0],torch.zeros_like(mask),True) is x
        names.append(str(dtype)+': off identity, no target identity, fixed path parity, reduced bound, dominant preservation, row/column isolation, no input mutation')
    x=torch.tensor([[2.,1.,0.,-40.,-40.]],dtype=torch.float64)
    y=x.clone();y[0,3:]-=100
    mask=torch.tensor([[True,False]])
    a=reflection_margin_penalty(x,[0,1],mask,True)
    b=reflection_margin_penalty(y,[0,1],mask,True)
    assert torch.equal(a[:,:3],b[:,:3])
    masked=torch.tensor([[2.,-torch.inf]])
    assert torch.equal(reflection_margin_penalty(masked,[0],torch.tensor([[True]]),True),masked)
    names.extend(['tail perturbation invariance for unchanged maxima','no finite alternative leaves row unchanged'])
    return names


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_checks(),indent=2))
