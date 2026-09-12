"""CPU source audit of dynamic-parameter identity; no KV-cache behavior claim."""
import argparse
import ast
import copy
import math
from pathlib import Path
import sys
import types

from mechanism_candidates import ROOT, BASE, read, save, sha, require


def audit():
    request_path = ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/request.py'
    worker_path = request_path.with_name('worker_manager.py')
    tree = ast.parse(request_path.read_text(encoding='utf-8'))
    names = {'STEER_CLAUSE_FIELDS', 'STEER_APPLY_FIELDS',
             'STEER_MOE_FIELDS', 'STEER_REBALANCE_FIELDS'}
    registry = [node for node in tree.body if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name) and node.target.id in names]
    namespace = {}
    exec(compile(ast.Module(body=registry, type_ignores=[]), str(request_path), 'exec'), namespace)
    worker_tree = ast.parse(worker_path.read_text(encoding='utf-8'))
    function = next(node for node in worker_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'config_fingerprint')
    namespace['SteerVectorRequest'] = object
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker_path), 'exec'), namespace)
    # An in-memory vector deliberately avoids file-version I/O. The real
    # fingerprint function imports file_version but does not call it here.
    module_name = 'vllm.steer_vectors.store'
    old = sys.modules.get(module_name)
    stub = types.ModuleType(module_name)
    def reject_file_version(path):
        raise AssertionError('Unexpected file-version lookup in in-memory fixture')
    stub.file_version = reject_file_version
    sys.modules[module_name] = stub
    try:
        params = read(ROOT/'.codex_work/question_balanced_20260911/original_selected_layer/fit.json')['parameters']
        fields = namespace['STEER_REBALANCE_FIELDS']
        values = dict(local_path=None, payload_sha256='fixed-payload', scale=1.0,
            target_layers=[20], apply_spec=None, algorithm='rebalance',
            normalize=False, is_multi_vector=False)
        values.update({name: params.get(name.removeprefix('rebalance_'), 0) for name in fields})
        values.update(rebalance_boundary_token_ids=[271], rebalance_think_start_token_id=151648,
            rebalance_think_end_token_id=151649, rebalance_paper_parameters=False,
            rebalance_negative_only=False, rebalance_sampled_confidence=False)
        request = types.SimpleNamespace(**values)
        fingerprint = namespace['config_fingerprint']
        reference = fingerprint(request); rows = []
        for field in fields:
            modified = copy.deepcopy(request)
            value = getattr(modified, field)
            changed = not value if isinstance(value, bool) else (
                value + [272] if isinstance(value, list) else (
                    value + 1 if isinstance(value, int) else math.nextafter(value, 0.0)))
            require(changed != value, 'Fixture field did not change')
            setattr(modified, field, changed)
            rows.append(dict(field=field, changes_fingerprint=fingerprint(modified) != reference))
        control = copy.deepcopy(request); control.scale = 2.0
        require(fingerprint(control) != reference, 'Positive control did not change identity')
    finally:
        if old is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = old
    evaluator = ROOT/BASE/'eval/rebalance_dynamic_eval.py'
    evaluator_tree = ast.parse(evaluator.read_text(encoding='utf-8'))
    settings = [kw.value.value for node in ast.walk(evaluator_tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'LLM'
                for kw in node.keywords if kw.arg == 'enable_prefix_caching'
                and isinstance(kw.value, ast.Constant)]
    require(settings == [False], 'Research entry prefix-caching policy changed')
    cache = ROOT/'sources/EasySteer/vllm-steer/vllm/v1/core/kv_cache_utils.py'
    return dict(status='CPU_dynamic_configuration_identity_audited_no_generation',
        function='worker_manager.config_fingerprint', fields=rows,
        dynamic_fields_omitted=sum(not row['changes_fingerprint'] for row in rows),
        scale_positive_control=True, current_research_enable_prefix_caching=False,
        finding='The real fingerprint ignores these dynamic ReBalance fields. The same function is used by prefix-cache key construction.',
        limits='Configuration identity only. This does not demonstrate an erroneous cache hit, output change, slot-routing bug, or damage to frozen results. Current research disables prefix caching; per-request dynamic state can coexist with shared vector slots.',
        decision='Record a limitation for future dynamic prefix-cache support. Do not enable caching or alter the fixed candidate batch.',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p, source=True)
                       for p in (request_path, worker_path, cache, evaluator, Path(__file__))},
        model_loads=0, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable audit exists')
    result = audit(); save(args.output, result)
    print({key:result[key] for key in ('status', 'dynamic_fields_omitted',
                                      'current_research_enable_prefix_caching', 'GPU_calls')})


if __name__ == '__main__':
    main()
