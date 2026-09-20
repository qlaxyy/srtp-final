"""Check the actual mixture formula using a CPU NumPy adapter; no forward."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/sample/hybrid_termination.py'


class Tensor(np.ndarray):
    def __new__(cls, x):
        return np.asarray(x).view(cls)

    def float(self):
        return Tensor(self.astype(np.float32))


def softmax(x):
    a = np.exp(x-x.max(axis=-1,keepdims=True))
    return a/a.sum(axis=-1,keepdims=True)


def main():
    fn = next(n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='mixed_end_column')
    def check(x, message):
        if not x: raise ValueError(message)
    ns = dict(math=math,torch=SimpleNamespace(
        logsumexp=lambda x,dim:np.logaddexp.reduce(x,axis=dim),
        logaddexp=np.logaddexp,isfinite=np.isfinite,_assert_async=check))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
    rng=np.random.default_rng(20260913);max_error=0.
    for _ in range(200):
        x=Tensor(rng.uniform(-30,30,(8,17)))
        x[rng.random(x.shape)<.3]=-np.inf;x[:,0]=0
        p=softmax(x);y=x.copy();y[:,-1]=ns['mixed_end_column'](x,16)
        actual=softmax(y);target=.95*p;target[:,-1]+=.05
        max_error=max(max_error,float(abs(actual-target).max()))
        assert np.allclose(actual,target,atol=3e-6)
        assert np.array_equal(x[:,:-1],y[:,:-1])
        assert np.all(.5*abs(actual-p).sum(-1)<=.050004)
    x=Tensor([[0.,-np.inf]])
    y=x.copy();y[:,1]=ns['mixed_end_column'](x,1)
    assert np.allclose(softmax(y),[[.95,.05]],atol=1e-7)
    try:ns['mixed_end_column'](Tensor([[-np.inf,-np.inf]]),1)
    except ValueError:pass
    else:raise AssertionError('Empty support accepted')
    result=dict(status='pass_cpu_formula_native_torch_pending',
                random_rows=1600,max_probability_error=max_error,
                filtered_end_probability_example=.05,non_end_logits_unchanged=True,
                single_step_TV_bound=.05,extra_model_forwards=0,
                source_sha256=hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                limitations='NumPy adapter, not Torch/BF16/RNG validation; per-step TV is not an accuracy bound')
    with (HERE/'cpu_checks.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
