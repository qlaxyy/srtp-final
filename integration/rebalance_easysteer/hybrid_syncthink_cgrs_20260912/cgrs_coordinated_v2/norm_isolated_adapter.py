"""Experimental opaque correction branch, preserving the original additive chain."""
import torch


@torch.library.custom_op('hybrid_norm::correction', mutates_args=('counters',))
def correction(hidden: torch.Tensor, residual: torch.Tensor, vectors: torch.Tensor,
               mask: torch.Tensor, rows: torch.Tensor, enabled: torch.Tensor,
               counters: torch.Tensor) -> torch.Tensor:
    n=hidden.shape[0];active=((mask[:n]!=0)&(rows[:n]>0)&(enabled[0]!=0))[:,None]
    x=hidden.float()+residual.float()
    old=hidden+mask[:n,None]*vectors[rows[:n]]
    y=old.float()+residual.float()
    nx=x.norm(dim=-1,keepdim=True);ny=y.norm(dim=-1,keepdim=True)
    valid=(nx>0)&(ny>0)&torch.isfinite(nx)&torch.isfinite(ny)
    target=y*(nx/torch.where(valid,ny,torch.ones_like(ny)))
    delta=torch.where(valid,target-y,hidden.float()-old.float())
    counters.add_(torch.stack((active.sum(),(active&~valid).sum())).to(counters.dtype))
    return torch.where(active,delta,torch.zeros_like(delta))


@correction.register_fake
def _(hidden,residual,vectors,mask,rows,enabled,counters):
    return torch.empty_like(hidden,dtype=torch.float32)


def make_isolated_kernel(original,enabled,counters):
    def kernel(tables,graph_mask,replace_mask,normalize_flag,token_rows,hidden_states,residual):
        assert set(tables)=={'additive'}
        old=original(tables,graph_mask,replace_mask,normalize_flag,token_rows,hidden_states,residual)
        delta=correction(hidden_states,residual if residual is not None else torch.zeros_like(hidden_states),
            tables['additive']['V'],graph_mask,token_rows,enabled,counters)
        return (old+delta).to(old.dtype)
    return kernel
