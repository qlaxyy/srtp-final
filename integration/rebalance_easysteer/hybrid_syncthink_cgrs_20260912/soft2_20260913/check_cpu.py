"""CPU algebra and AST-extracted production callback, without Torch or vLLM.

Contract: post-filter end-only bias preserves all surviving alternatives and
never revives excluded tokens. Numpy adapter checks callback indexing and
state, not Torch/CUDA kernels. Actual runtime tests remain a GPU-entry gate.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from build_patch import variants, HERE


class Tensor(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    is_cuda = False

    def float(self):
        return Tensor(self, np.float32)

    def long(self):
        return Tensor(self, np.int64)

    def to(self, dtype):
        return Tensor(self, dtype)


def softmax(x):
    w = np.exp(x - np.max(x))
    return w / w.sum()


def main():
    old, new = variants()
    source = next(t for n, t in new.items() if n.endswith('hybrid_termination.py'))
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'apply_after_filter')
    ns = dict(math=math, torch=SimpleNamespace(
        isfinite=np.isfinite,
        where=lambda c, a, b: Tensor(np.where(c, a, b))))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<callback>', 'exec'), ns)
    apply = ns['apply_after_filter']
    cases = 0
    for dtype in (np.float32, np.float16):
        logits = Tensor(np.full((5, 151650), -np.inf, dtype=dtype))
        # Mixed rows, slot permutation, filtered end, singleton end, untriggered.
        logits[:, 3] = 0
        logits[[0, 2, 3, 4], 151649] = -1
        logits[3, 3] = -np.inf
        before = logits.copy()
        state = SimpleNamespace(
            pending_soft=(Tensor([2, 0, 1, 3]), Tensor([4, 1, 3, 0]),
                          Tensor([True, True, True, False])),
            first_bias=Tensor([-1]*5), count=Tensor([5, 6, 7, 8, 9]),
            bias_count=Tensor([0]*5), filtered_trigger_count=Tensor([0]*5),
            events=[])
        assert apply(state, logits) is logits
        assert np.array_equal(np.isfinite(before), np.isfinite(logits))
        assert np.array_equal(before[:, :151649], logits[:, :151649])
        assert np.array_equal(before[[1, 3, 4]], logits[[1, 3, 4]])
        assert state.first_bias.tolist() == [-1, 6, -1, -1, 9]
        assert state.filtered_trigger_count.tolist() == [0, 0, 0, 1, 0]
        assert state.bias_count.tolist() == [0, 1, 0, 0, 1]
        after = logits.copy()
        assert apply(state, logits) is logits and np.array_equal(after, logits)
        # Float rounding is measured, not assumed exactly log(2) in BF16.
        observed = float(logits[0,151649] - before[0,151649])
        assert abs(observed - math.log(2)) < .001
        cases += 1
    # Analytic distribution: q' = 2q/(1+q), max single-step TV is 3-2sqrt(2).
    q = np.linspace(0, 1, 10001)
    qp = 2*q/(1+q)
    assert max(qp-q) <= 3-2*math.sqrt(2)+1e-12
    assert np.all(qp[:-1] < 1) and qp[0] == 0
    # Reweighting does not redistribute mass among continuation tokens.
    rng = np.random.default_rng(20260913)
    for _ in range(200):
        x = rng.normal(size=25)
        x[rng.random(25) < .3] = -np.inf
        x[0] = 0
        p = softmax(x)
        y = x.copy(); y[-1] += math.log(2)
        z = softmax(y)
        assert np.array_equal(p > 0, z > 0)
        assert np.allclose(z[-1], 2*p[-1]/(1+p[-1]))
        assert np.allclose(p[:-1]/p[:-1].sum(), z[:-1]/z[:-1].sum())
    # Actual sampler AST: hook is strictly after filtering and before gumbel;
    # strip only the new optional hook to prove the default body is identical.
    sampler_name = next(n for n in new if n.endswith('/sampler.py'))
    def sample_ast(text):
        return next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == 'sample')
    before = sample_ast(old[sampler_name]); after = sample_ast(new[sampler_name])
    positions = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(after)
                 if isinstance(n, ast.Call)]
    line = lambda name: next(i for i, f in positions if f == name)
    assert line('apply_top_k_top_p') < line('after_filter') < line('gumbel_sample')
    class RemoveHook(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == 'after_filter is not None':
                return None
            return self.generic_visit(node)
    after = RemoveHook().visit(after)
    after.args.args = after.args.args[:-1]
    after.args.defaults = after.args.defaults[:-1]
    assert ast.dump(before) == ast.dump(after)
    # ReBalance reads original statistics before either intervention.
    runner = next(t for n, t in new.items() if n.endswith('model_runner.py'))
    assert runner.index('max_probabilities = torch.exp(') < runner.index(
        'hybrid_ticket = hybrid.read_apply(') < runner.index('after_filter=hybrid.')
    # Syntax and patch anchoring checked without importing model code.
    for name, text in new.items():
        ast.parse(text, feature_version=(3, 10))
    result = dict(status='cpu_spec_pass_runtime_pending',
                  callback_dtypes=['numpy float32', 'numpy float16'],
                  callback_cases=cases, distribution_cases=200,
                  probability_grid_points=len(q),
                  support_preserved=True, non_end_relative_weights_preserved=True,
                  default_sampler_ast_identical_after_removing_optional_hook=True,
                  hook_after_temperature_and_top_p=True,
                  R_statistics_before_intervention=True,
                  max_exact_single_step_TV=3-2*math.sqrt(2),
                  extra_forwards=0, server_connections=0,
                  limitations=['Numpy adapter is not native Torch/CUDA validation',
                               'No BF16, scheduling, RNG or latency validation',
                               'Single-step bound does not bound trajectory accuracy',
                               'No new generated answers or estimated new accuracy'])
    if (HERE/'cpu_checks.json').exists():
        assert json.loads((HERE/'cpu_checks.json').read_text(encoding='utf-8')) == result
    else:
        with (HERE/'cpu_checks.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
