"""Audit the adapter against executable author controller code on CPU only.

No model or vLLM package import is required. Reports differences, not a pass
claim; upstream extraction intentionally fails if the source structure changes.
"""

import ast
import hashlib
import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path

import torch

if '--cuda' in sys.argv:
    torch.set_default_device('cuda')

root = Path(__file__).resolve().parents[3]
author_path = root / 'sources/ReBalance/modeling_utils/modeling_qwen2_dynamic_3D.py'
adapter_path = root / 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
source = author_path.read_text(encoding='utf-8')
tree = ast.parse(source)
namespace = {'torch': torch, '_m': math}
for name in ('_solve_k_for_tau', 'build_F'):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1
    exec(compile(ast.Module(body=matches, type_ignores=[]), str(author_path), 'exec'), namespace)
spec = importlib.util.spec_from_file_location('audit_rebalance_adapter', adapter_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
hp = dict(q25c=0.662293, q75c=0.94805, low_val_1=-1.02, q25v=0.000560,
          q75v=0.011597, low_val_2=-1.91, high_val_2=0.1)
params = module.ReBalanceParams(boundary_token_ids=(99,), think_start_token_id=10,
                               think_end_token_id=11, **hp)
from types import SimpleNamespace
block = source.split('# —— dynamic steering hyperparameters')[1].split('\n', 1)[1]
block = block.split('self._coefs[ready_mask] = updated')[0]
block = textwrap.dedent(block)

def compare(c, v):
    namespace.update(self=SimpleNamespace(_dyn_hparams=hp), mean_max=c, var_conf=v)
    exec(compile(block, str(author_path), 'exec'), namespace)
    author = namespace['updated']
    adapted = module.compute_rebalance_coefficient(c, v, params)
    return author, adapted

c = torch.tensor([0.7, 0.8, 0.9, hp['q25c'], 1.0], dtype=torch.float32)
v = torch.tensor([0.001, 0.001, 0.001, hp['q75v'], hp['q25v']])
a, b = compare(c, v)
cg, vg = torch.meshgrid(torch.linspace(0, 1, 1001), torch.linspace(0, 0.25, 251), indexing='ij')
ag, bg = compare(cg.flatten(), vg.flatten())
diff = (ag-bg).abs()
index = int(diff.argmax())
result = dict(scope='function audit, no model generation', device=str(c.device), torch=torch.__version__,
    author_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    adapter_source_sha256=hashlib.sha256(adapter_path.read_text(encoding='utf-8').encode()).hexdigest(),
    examples=[dict(c=float(ci), v=float(vi), author=float(ai), adapter=float(bi)) for ci,vi,ai,bi in zip(c,v,a,b)],
    grid_points=diff.numel(), max_abs_error=float(diff.max()),
    worst_c=float(cg.flatten()[index]), worst_v=float(vg.flatten()[index]),
    mismatches_at_1e_5=int((diff>1e-5).sum()),
    sign_mismatches=int(((ag<0)&(bg>0) | (ag>0)&(bg<0)).sum()))
print(json.dumps(result, indent=2))
if '--assert-equivalent' in sys.argv:
    torch.testing.assert_close(ag, bg, atol=1e-5, rtol=1e-5)
