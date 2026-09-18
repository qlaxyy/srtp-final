"""CPU reference only, not installed into vLLM; complete-state norm correction."""
import torch


def transform(hidden, residual, vector, scales, *, enabled=False):
    # Keep original arithmetic when disabled; no norm reductions on this branch.
    old = hidden + scales.to(hidden.dtype).unsqueeze(-1) * vector
    if not enabled:
        return old
    x = hidden.float()
    if residual is not None:
        x = x + residual.float()
    # Start from the same dtype-rounded addition as the existing ReBalance path.
    y = old.float()
    if residual is not None:
        y = y + residual.float()
    nx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ny = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    usable = (nx > 0) & (ny > 0) & torch.isfinite(nx) & torch.isfinite(ny)
    scale = nx / torch.where(usable, ny, torch.ones_like(ny))
    z = y * scale
    active = usable & (scales.unsqueeze(-1) != 0)
    corrected = (old.float() + z - y).to(hidden.dtype)
    # Degenerate states fall back to the original hidden state, not NaN/Inf.
    return torch.where(active, corrected, hidden)


def check():
    torch.manual_seed(42)
    reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        hidden=torch.randn(64,1536).to(dtype)
        residual=torch.randn_like(hidden)
        v=torch.randn(1536).to(dtype)
        a=torch.linspace(-1.535345,.1,64)
        old=hidden+a.to(dtype)[:,None]*v
        assert torch.equal(transform(hidden,residual,v,a),old)
        zero=transform(hidden,residual,v,torch.zeros(64),enabled=True)
        assert torch.equal(zero,hidden)
        residual_before=residual.clone()
        out=transform(hidden,residual,v,a,enabled=True)
        assert torch.equal(residual,residual_before)
        before=torch.linalg.vector_norm(hidden.float()+residual.float(),dim=-1)
        after=torch.linalg.vector_norm(out.float()+residual.float(),dim=-1)
        err=float(((after-before).abs()/before).max())
        assert err < (1e-6 if dtype==torch.float32 else .002)
        assert torch.isfinite(out).all()
        reports.append(dict(dtype=str(dtype),max_relative_complete_norm_error=err,
                            disabled_exact=True,zero_exact=True,residual_unchanged=True))
    h=torch.ones(1,4)
    assert torch.equal(transform(h,None,-h[0],torch.ones(1),enabled=True),h)
    return reports


if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
