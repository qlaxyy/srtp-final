"""Replay reviewed fused-kernel arithmetic without loading a language model."""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_kernel(artifact):
    raw=artifact.read_bytes()
    segments=re.finditer(rb'[\x09\x0a\x0d\x20-\x7e]{1000,}',raw)
    text=next(m.group().decode('ascii') for m in segments
              if b'Compile-time auto-tuning block:' in m.group())
    kernels=re.findall(r"= async_compile\.triton\('[^']+', '''(.*?)''', device_str='cuda'\)",text,re.S)
    selected=[k for k in kernels if 'def triton_red_fused__to_copy' in k]
    assert len(selected)==2 and selected[0]==selected[1]
    source=selected[0][selected[0].index('@triton.jit'):]
    source=re.sub(r'def triton_red_fused_\w+\(', 'def replay_kernel(',source,count=1)
    return ('import triton\nimport triton.language as tl\n'
            'from torch._inductor.runtime import triton_helpers\n'
            'from torch._inductor.runtime.triton_helpers import libdevice\n\n'+source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists()
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True)
    kernels={}
    for label in ['additive','feedback']:
        source=a.artifacts/f'{label}_subgraph1.artifact'
        assert digest(source)==plan['artifacts'][label]['sha256']
        p=a.output/f'{label}_replay_kernel.py'
        p.write_text(extract_kernel(source),encoding='utf-8')
        assert digest(p)==plan['reviewed_kernel_source_sha256'][p.name]
        spec=importlib.util.spec_from_file_location(label+'_kernel',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        kernels[label]=module.replay_kernel
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU process'
    import torch
    torch.set_num_threads(4);torch.manual_seed(plan['seed'])
    start=time.perf_counter()
    vector_path=a.assets/'auto_vector.pt';readout_path=a.assets/'readout.npy'
    assert digest(vector_path)=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'
    assert digest(readout_path)=='4d278704b4ba314bf332b5f2609fd979c7cf266207c0a1f5931beb0dd0d53a1e'
    import numpy as np
    direction=torch.load(vector_path,map_location='cpu',weights_only=True)
    if isinstance(direction,dict):direction=direction[next(iter(direction))]
    direction=torch.as_tensor(direction).flatten().to('cuda',torch.bfloat16)
    readout=torch.from_numpy(np.load(readout_path)).flatten().to('cuda',torch.float32)
    width=plan['width'];assert direction.numel()==readout.numel()==width
    d=torch.zeros(129,width,device='cuda',dtype=torch.bfloat16);d[1]=direction
    w=torch.zeros(129,width,device='cuda');w[1]=readout
    enabled=torch.zeros(129,1,device='cuda');center=torch.zeros_like(enabled)
    normalize=torch.zeros(129,device='cuda',dtype=torch.bfloat16)
    records=[]
    def metrics(left,right):
        error=left.float()-right.float()
        return {'equal':bool(torch.equal(left,right)),
                'different_elements':int(torch.count_nonzero(error)),
                'max_abs':float(error.abs().max()),
                'rms':float(error.square().mean().sqrt())}
    for n in plan['rows']:
        h=(torch.randn(n,width)*4).bfloat16().cuda()
        r1=(torch.randn(n,width)*3).bfloat16().cuda()
        r2=(torch.randn(n,width)*3).bfloat16().cuda()
        weight=(torch.randn(width)*.1+1).bfloat16().cuda()
        rows=torch.ones(n,device='cuda',dtype=torch.long)
        for mask_name in plan['masks']:
            c=(torch.zeros(n) if mask_name=='zero' else torch.linspace(-1.5,.1,n)).bfloat16().cuda()
            out=lambda dtype=torch.bfloat16: torch.empty((n,width),device='cuda',dtype=dtype)
            additive_residual=out();additive_norm=out()
            kernels['additive'][(n,)](h,r1,r2,c,rows,d,normalize,weight,
                out(),out(torch.float32),additive_residual,additive_norm,n,width,**plan['launch'])
            record={'rows':n,'mask':mask_name,'arms':{}}
            for storage_name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
                buffer=h.to(dtype).clone();feedback_residual=out();feedback_norm=out()
                kernels['feedback'][(n,)](buffer,r1,r2,rows,w,d,c,enabled,center,
                    normalize,weight,torch.empty(n,1,device='cuda',dtype=torch.bfloat16),
                    out(),feedback_residual,feedback_norm,n,width,**plan['launch'])
                torch.cuda.synchronize()
                record['arms'][storage_name]={'normalized':metrics(additive_norm,feedback_norm),
                    'residual':metrics(additive_residual,feedback_residual)}
            records.append(record)
    supported=all(r['arms']['bf16']['normalized']['different_elements']>0 and
        r['arms']['fp32']['normalized']['rms']<=r['arms']['bf16']['normalized']['rms']*.01
        for r in records if r['mask']!='zero')
    result={'plan_sha256':digest(a.plan),'script_sha256':digest(Path(__file__)),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'torch_version':torch.__version__,'records':records,'supports_local_kernel_mechanism':supported,
            'seconds':time.perf_counter()-start,'model_calls':0,'new_answers':0,
            'limitations':plan['limitations']}
    (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
