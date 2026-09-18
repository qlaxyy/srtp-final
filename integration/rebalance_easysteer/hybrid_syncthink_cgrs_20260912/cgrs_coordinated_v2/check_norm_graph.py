"""Synthetic graph ABI checks: dynamic signed masks, off identity, reset, residual."""
import importlib.util,json,sys
from pathlib import Path
import torch
from norm_graph_adapter import make_kernel

def check(kernel_path,device='cpu'):
    spec=importlib.util.spec_from_file_location('isolated_graph_kernels',kernel_path)
    m=importlib.util.module_from_spec(spec)
    # Local CPU Python is 3.9; defer type annotations without changing kernel math.
    exec(compile('from __future__ import annotations\n'+Path(kernel_path).read_text(encoding='utf8'),str(kernel_path),'exec'),m.__dict__)
    torch.manual_seed(42);reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        h=torch.randn(8,1536,device=device,dtype=dtype)
        r=torch.randn_like(h);before=r.clone()
        table=torch.randn(3,1536,device=device,dtype=dtype);table[0].zero_()
        mask=torch.tensor([0,-1,-1.535345,.1,0,0,1,0],device=device,dtype=dtype)
        rows=torch.tensor([0,1,1,2,1,0,2,0],device=device)
        args=({'additive':{'V':table}},mask,torch.zeros_like(mask),torch.zeros(3,device=device,dtype=dtype),rows,h,r)
        mode=torch.zeros(1,device=device,dtype=torch.int32)
        count=torch.zeros(2,device=device,dtype=torch.int64)
        fn=make_kernel(m.apply_decoder_families,mode,count)
        old=m.apply_decoder_families(*args)
        assert torch.equal(fn(*args),old) and count.sum()==0
        mode.fill_(1);out=fn(*args);active=(mask!=0)&(rows>0)
        assert torch.equal(out[~active],old[~active]) and torch.equal(r,before)
        nx=(h.float()+r.float()).norm(dim=1);ny=(out.float()+r.float()).norm(dim=1)
        error=float(((ny[active]-nx[active]).abs()/nx[active]).max())
        assert error<(1e-6 if dtype==torch.float32 else .002)
        assert count.tolist()==[4,0]
        mode.zero_();count.zero_();assert torch.equal(fn(*args),old)
        reports.append(dict(dtype=str(dtype),device=device,relative_norm_error=error,off_exact=True,masked_exact=True,mode_reset=True))
    return reports

if __name__=='__main__':
    print(json.dumps(check(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'cpu'),indent=2))
