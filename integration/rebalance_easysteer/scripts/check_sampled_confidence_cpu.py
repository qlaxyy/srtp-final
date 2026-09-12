"""Run actual confidence/state/runner functions with a minimal NumPy backend.

This validates local arithmetic, ownership and routing, not Torch/CUDA execution.
The corresponding vendor tests must still run in the existing server environment.
"""
import argparse
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require

VENDOR = ROOT/'sources/EasySteer/vllm-steer'


class Array(np.ndarray):
    @property
    def device(self): return 'cpu'
    def __getitem__(self, key):
        value=super().__getitem__(key)
        result=np.asarray(value).view(Array)
        if not isinstance(value,np.ndarray): result._scalar_parent=(self,key)
        return result
    def clone(self): return arr(np.array(self,copy=True))
    def detach(self): return self
    def cpu(self): return self
    def to(self, dtype=None, copy=False): return np.array(self, dtype=dtype, copy=copy).view(Array)
    def float(self): return self.astype(np.float32, copy=False)
    def long(self): return self.astype(np.int64, copy=False)
    def clamp(self, min=None, max=None): return arr(np.clip(self, min, max))
    def clamp_min(self, value): return self.clamp(min=value)
    def square(self): return arr(np.square(self))
    def fill_(self, value): return self.copy_(value)
    def zero_(self): return self.fill_(0)
    def copy_(self, value):
        self[...] = value
        if hasattr(self,'_scalar_parent'):
            parent,key=self._scalar_parent;parent[key]=value
        return self
    def index_select(self, dim, indices): return arr(np.take(self, indices, axis=dim))
    def index_copy_(self, dim, indices, values):
        assert dim == 0
        self[indices] = values
        return self
    def amax(self, dim): return arr(np.max(np.asarray(self), axis=dim))
    def any(self, dim=None): return arr(np.any(np.asarray(self), axis=dim))
    def gather(self, dim, indices): return arr(np.take_along_axis(self, indices, axis=dim))
    def squeeze(self, dim): return arr(np.squeeze(np.asarray(self), axis=dim))
    def new_zeros(self, shape, dtype=None): return arr(np.zeros(shape, dtype=dtype))


def arr(value, dtype=None, device='cpu'):
    require(device == 'cpu', 'Local backend cannot represent a GPU')
    return np.asarray(value, dtype=dtype).view(Array)


def logsumexp(value, dim):
    value = np.asarray(value)
    maximum = np.max(value, axis=dim, keepdims=True)
    return arr(np.squeeze(maximum, axis=dim) + np.log(np.exp(value-maximum).sum(axis=dim)))


def backend():
    def tensor(value, dtype=None, device='cpu'):
        if dtype is None and not isinstance(value,np.ndarray) and np.asarray(value).dtype.kind=='f':
            dtype=np.float32
        return arr(value,dtype=dtype,device=device)
    def shape_args(*shape): return shape[0] if len(shape)==1 else shape
    def zeros(*shape, dtype=np.float32, device='cpu'):
        return arr(np.zeros(shape_args(*shape), dtype=dtype), device=device)
    def ones(*shape, dtype=np.float32, device='cpu'):
        return arr(np.ones(shape_args(*shape), dtype=dtype), device=device)
    def assert_close(a, b, equal_nan=False, **kwargs):
        np.testing.assert_allclose(a, b, rtol=kwargs.get('rtol',1e-5), atol=kwargs.get('atol',1e-7), equal_nan=equal_nan)
    return SimpleNamespace(Tensor=Array, tensor=tensor,
        device=lambda name:name, float32=np.float32, long=np.int64, bool=np.bool_, int32=np.int32,
        zeros=zeros, ones=ones, zeros_like=lambda v:arr(np.zeros_like(v)),
        full_like=lambda v, fill:arr(np.full_like(v,fill)),
        arange=lambda *args, **kw:arr(np.arange(*args,dtype=kw.get('dtype')),device=kw.get('device','cpu')),
        exp=lambda v:arr(np.exp(v)), tanh=lambda v:arr(np.tanh(v)),
        logsumexp=logsumexp, nan_to_num=lambda v,**kw:arr(np.nan_to_num(v,**kw)),
        where=lambda c,a,b:arr(np.where(c,a,b)), isfinite=lambda v:arr(np.isfinite(v)),
        sigmoid=lambda v:arr(1/(1+np.exp(-v))), nan=np.nan, no_grad=nullcontext,
        testing=SimpleNamespace(assert_close=assert_close))


def execute_nodes(path, names, namespace, method=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if method:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
            isinstance(f, ast.FunctionDef) and f.name==method for f in n.body))
        nodes = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name==method]
    else:
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        require({n.name for n in nodes} == set(names), 'Missing extracted functions')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def check(output):
    started = time.perf_counter()
    paths = dict(controller=VENDOR/'vllm/steer_vectors/rebalance.py',
        state=VENDOR/'vllm/v1/worker/gpu/steer_vector_utils.py',
        runner=VENDOR/'vllm/v1/worker/gpu/model_runner.py',
        tests=VENDOR/'tests/steer_vectors/test_rebalance.py')
    ns = dict(__name__=__name__, torch=backend(), np=np, math=math, dataclass=dataclass,
              lru_cache=lru_cache, SimpleNamespace=SimpleNamespace, unittest=unittest)
    names = ['ReBalanceParams','snapshot_raw_confidence','sampled_raw_confidence',
        '_solve_k_for_tau','validate_curve_targets','_baseline','_curve_constants',
        'compute_rebalance_coefficient','compute_paper_coefficient']
    execute_nodes(paths['controller'], names, ns)
    execute_nodes(paths['state'], ['SteerVectorState'], ns)
    execute_nodes(paths['runner'], [], ns, method='sample')
    execute_nodes(paths['tests'], ['_request','_Manager','_replay_batch',
        'test_sampled_confidence_is_request_local_across_reordering_and_replay',
        'test_rebalance_state_is_request_local_and_uses_arithmetic_mean'], ns)
    checks = []
    for name in ['test_rebalance_state_is_request_local_and_uses_arithmetic_mean',
                 'test_sampled_confidence_is_request_local_across_reordering_and_replay']:
        ns[name](); checks.append(name)
    rng = np.random.default_rng(20260926)
    tested_rows = 0; snapshot_calls = []
    actual_snapshot = ns['snapshot_raw_confidence']
    def snapshot(logits):
        snapshot_calls.append(1)
        return actual_snapshot(logits)
    ns['snapshot_raw_confidence'] = snapshot
    for n in (1, 2, 7, 128):
        for selected in (False, True):
            for grammar in (False, True):
                logits = arr(rng.normal(0, 3, (n, 37)), dtype=np.float32)
                raw = np.array(logits)
                ids = arr(rng.integers(1, 37, (n, 1)), dtype=np.int32)
                observed = []; mutations = []
                state = SimpleNamespace(has_dynamic=lambda:True,
                    requires_confidence=lambda:True, requires_sampled_confidence=lambda:selected,
                    observe_sample=lambda *args:observed.append(args))
                def mask(value, *args): value[:,0]=-np.inf; mutations.append('grammar')
                def sampler(value, batch):
                    value *= 1/.7
                    value[:,1] -= 3
                    mutations.append('sampler')
                    return SimpleNamespace(sampled_token_ids=ids,num_sampled=None,num_rejected=None)
                runner = SimpleNamespace(model=SimpleNamespace(compute_logits=lambda hidden:logits),
                    steer_vector_state=state, sampler=sampler,rejection_sampler=None,
                    structured_outputs_worker=SimpleNamespace(apply_grammar_bitmask=mask))
                batch = SimpleNamespace(logits_indices=arr(np.arange(n)), num_draft_tokens=0,num_reqs=n)
                before = len(snapshot_calls)
                ns['sample'](runner,arr(np.zeros((n,2))),batch,
                    SimpleNamespace(structured_output_request_ids=[],grammar_bitmask=None) if grammar else None)
                require(len(snapshot_calls)-before == int(selected), 'Disabled snapshot allocated')
                _, returned_ids, maxima, chosen = observed[0]
                require(np.array_equal(ids,returned_ids), 'Sampler result changed')
                reference = np.exp(raw-raw.max(axis=1,keepdims=True))
                reference /= reference.sum(axis=1,keepdims=True)
                np.testing.assert_allclose(maxima,reference.max(axis=1),rtol=2e-6,atol=1e-7)
                if selected:
                    expected = np.take_along_axis(reference,ids,axis=1).ravel()
                    np.testing.assert_allclose(chosen,expected,rtol=2e-6,atol=1e-7)
                else: require(chosen is None,'Default consumes selected probabilities')
                require(mutations==(['grammar','sampler'] if grammar else ['sampler']),'Mutation ordering changed')
                tested_rows += n
    checks.append('actual_runner_16_mutating_sampler_cases')
    old_source=subprocess.check_output(['git','show',
        'ddb3cd2:sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/model_runner.py'],cwd=ROOT).decode()
    old_class=next(n for n in ast.parse(old_source).body if isinstance(n,ast.ClassDef) and any(
        isinstance(f,ast.FunctionDef) and f.name=='sample' for f in n.body))
    old_sample=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
    old_ns=dict(ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),old_sample],
        type_ignores=[])), '<frozen-pre-candidate-runner>', 'exec'),old_ns)
    for confidence in (True,False):
        saved=[]
        for sample in (old_ns['sample'],ns['sample']):
            logits=arr([[2.,-1.,3.],[0.,-4.,.5]],dtype=np.float32)
            observed=[]
            state=SimpleNamespace(has_dynamic=lambda:True,
                requires_confidence=lambda:confidence, requires_sampled_confidence=lambda:False,
                observe_sample=lambda *args:observed.append(args))
            def sampler(value,batch):
                value *= .7
                return SimpleNamespace(sampled_token_ids=arr([[0],[2]]),num_sampled=None,num_rejected=None)
            runner=SimpleNamespace(model=SimpleNamespace(compute_logits=lambda hidden:logits),
                steer_vector_state=state,sampler=sampler,rejection_sampler=None)
            batch=SimpleNamespace(logits_indices=arr([0,1]),num_draft_tokens=0,num_reqs=2)
            sample(runner,arr(np.zeros((2,2))),batch,None)
            saved.append((np.array(observed[0][2]),np.array(logits)))
        require(all(np.array_equal(a,b) for a,b in zip(*saved)),'Disabled runner changed numerically')
    checks.append('default_and_constant_runner_exact_prechange_equivalence')
    probe = arr([[2.,0.,-1.],[-3.,1.,2.]],dtype=np.float32)
    snap = actual_snapshot(probe); retained=np.array(probe);probe[:]=17
    require(not np.shares_memory(probe,snap[0]) and np.array_equal(snap[0],retained),'Raw snapshot aliases mutated FP32 logits')
    np.testing.assert_equal(ns['sampled_raw_confidence'](*snap,arr([[-1],[3]])),[0.,0.])
    for wrong in (arr([1,0]),arr([[1,2],[0,1]])):
        try:ns['sampled_raw_confidence'](*snap,wrong)
        except RuntimeError:pass
        else:raise AssertionError('Malformed sample shape accepted')
    checks.append('snapshot_ownership_dummy_rows_shape_rejection')
    greedy=arr(np.argmax(snap[0],axis=1)[:,None])
    require(np.array_equal(ns['sampled_raw_confidence'](*snap,greedy),
        ns['torch'].exp(snap[0].amax(dim=-1)-snap[1])),'Greedy selected and maximum confidence differ')
    checks.append('greedy_selected_equals_raw_max_exactly')
    for attribute,value in [('rebalance_sampled_confidence','true'),('rebalance_negative_only',True),
        ('algorithm','seal'),('algorithm','rebalance_radial'),('algorithm','rebalance_feedback'),
        ('rebalance_paper_parameters',[2.,2.,2.,.02,.002])]:
        request = ns['_request']();request.rebalance_sampled_confidence=True;setattr(request,attribute,value)
        try:ns['ReBalanceParams'].from_request(request)
        except ValueError:pass
        else:raise AssertionError('Unsupported combination accepted')
    checks.append('six_unsupported_configuration_rejections')
    result = dict(status='passed_local_numpy_backed_actual_source_checks', checks=checks,
        runner_rows=tested_rows, source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p,source=True) for p in paths.values()},
        checker_sha256=sha(Path(__file__),source=True),cpu_seconds=time.perf_counter()-started,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        tracked_worktree_dirty=bool(subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT).strip()),
        GPU_calls=0,model_loads=0,new_answers=0,
        limits='NumPy emulates the listed operations only. These are not Torch/CUDA or msgspec integration checks, throughput measurements, or generation benefit. Vendor tests and short real compiled checks are pending the next explicitly authorized GPU session.')
    if output:
        require(not output.exists(),'Immutable receipt exists'); save(output,result)
    print(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    check(parser.parse_args().output)
