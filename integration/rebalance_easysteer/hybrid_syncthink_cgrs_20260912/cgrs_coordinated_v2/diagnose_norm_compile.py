"""CUDA tensor-only diagnosis; no model weights, prompts or generated answers."""
import importlib.util,json,sys
from pathlib import Path
import torch
from norm_graph_adapter import make_kernel

spec=importlib.util.spec_from_file_location('kernel',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules['kernel']=m;spec.loader.exec_module(m)
torch.manual_seed(42);device='cuda';dtype=torch.bfloat16
h=torch.randn(32,1536,device=device,dtype=dtype)
r=torch.randn_like(h);v=torch.randn(3,1536,device=device,dtype=dtype);v[0].zero_()
mask=torch.tensor([0,-1,-1.535345,.1]*8,device=device,dtype=dtype)
rows=torch.tensor([0,1,1,2]*8,device=device)
args=({'additive':{'V':v}},mask,torch.zeros_like(mask),torch.zeros(3,device=device,dtype=dtype),rows,h,r)
enabled=torch.zeros(1,device=device,dtype=torch.int32);counts=torch.zeros(2,device=device,dtype=torch.int64)
old=m.apply_decoder_families;new=make_kernel(old,enabled,counts)
def downstream(fn):
    def f(*args):
        y=fn(*args)+r
        return y*torch.rsqrt((y.float()*y.float()).mean(-1,keepdim=True)+1e-6).to(y.dtype)
    return f
report=[]
for fused in (False,True):
    a=downstream(old) if fused else old;b=downstream(new) if fused else new
    ae=a(*args);be=b(*args)
    ac=torch.compile(a,fullgraph=True)(*args);bc=torch.compile(b,fullgraph=True)(*args)
    report.append(dict(downstream=fused,eager_equal=torch.equal(ae,be),compiled_equal=torch.equal(ac,bc),
       differing_elements=int((ac!=bc).sum()),max_abs_diff=float((ac.float()-bc.float()).abs().max()),
       original_compile_vs_eager=int((ac!=ae).sum()),wrapper_compile_vs_eager=int((bc!=be).sum())))
print(json.dumps(report,indent=2))
