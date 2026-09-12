"""Execute the actual CLI, fit-loading and controller-parameter source on CPU.

Only selected, explicitly bounded AST statements are executed. This does not
load a model, construct a Torch payload or validate the native CUDA runtime.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
import types

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from audit_control_alignment import load_controller
from check_prepared_prompts import FrozenByteBPE, TOKENIZER, TOKENIZER_SHA
from run_vector_batch import command

EVAL = ROOT / BASE / 'eval/rebalance_dynamic_eval.py'
CONTRACT = ROOT / BASE / 'eval/calibration_contract.py'


def assignment_name(node):
    target = (node.targets[0] if isinstance(node, ast.Assign) else
              node.target if isinstance(node, ast.AnnAssign) else None)
    return target.id if isinstance(target, ast.Name) else None


def compile_nodes(nodes, path):
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')


class AuditTokenizer(FrozenByteBPE):
    def get_vocab(self):
        return self.vocab | self.added

    def encode(self, text, add_special_tokens=False):
        require(not add_special_tokens, 'Unexpected special-token insertion')
        return super().encode(text)


def audit(bundle):
    parent = read(bundle / 'plan.json')
    tree = ast.parse(EVAL.read_text(encoding='utf-8'))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main = functions['main'].body
    stop = next(i for i, n in enumerate(main)
                if assignment_name(n) == 'model_path')
    prefix = main[:stop]
    selected_names = {'boundary_ids', 'think_start_id', 'think_end_id',
                      'dynamic_params'}
    parameters = [n for n in main if assignment_name(n) in selected_names]
    require({assignment_name(n) for n in parameters} == selected_names,
            'Controller source structure changed')
    switches = [n for n in main if isinstance(n, ast.If)
                and isinstance(n.test, ast.Attribute)
                and isinstance(n.test.value, ast.Name)
                and n.test.value.id == 'args'
                and n.test.attr in ('negative_only', 'sampled_confidence')]
    require(len(switches) == 2, 'Candidate switch source changed')
    result = next(n.value for n in main if assignment_name(n) == 'result')
    definition = next(v for k, v in zip(result.keys, result.values)
                      if isinstance(k, ast.Constant)
                      and k.value == 'confidence_definition')
    function_code = compile_nodes(
        [functions['parse_args'], functions['single_token_id']], EVAL)
    contract_code = compile_nodes(ast.parse(
        CONTRACT.read_text(encoding='utf-8')).body, CONTRACT)
    ns_base = dict(argparse=argparse, Path=Path, json=json, math=math,
                   AutoTokenizer=AuditTokenizer, DEFAULT_MODEL='',
                   DEFAULT_DATASET='', DEFAULT_VECTOR='', DEFAULT_OUTPUT='')
    exec(contract_code, ns_base)
    exec(function_code, ns_base)
    tokenizer = AuditTokenizer(TOKENIZER)
    module_name = 'vllm.steer_vectors.rebalance'
    module = types.ModuleType(module_name)
    module.validate_curve_targets = load_controller()['validate_curve_targets']
    missing = object(); previous = sys.modules.get(module_name, missing)
    old_argv = sys.argv[:]; records = []
    try:
        sys.modules[module_name] = module
        for candidate in parent['candidate_order']:
            for stage, entry in parent['candidates'][candidate].items():
                folder = bundle / entry['bundle']
                plan = read(folder / 'plan.json')
                require(sha(folder / 'plan.json') == entry['plan_sha256'],
                        'Child plan changed')
                for arm in plan['run_order']:
                    cmd = command(plan, folder, Path(plan['default_server_output']), arm)
                    sys.argv = cmd[2:]
                    ns = dict(ns_base, tokenizer=tokenizer)
                    # parse_args resolves its original global namespace; it only
                    # depends on argparse, Path and the deliberately unused defaults.
                    exec(compile_nodes(prefix, EVAL), ns)
                    exec(compile_nodes(parameters + switches, EVAL), ns)
                    args = ns['args']; actual = ns['dynamic_params']
                    fitted = read(args.calibration_fit)
                    require(sha(args.calibration_fit) == plan['arms'][arm]['fit_sha256'],
                            'Fit changed')
                    require(fitted['parameters'] == plan['dynamic_parameters'],
                            'Fit and declared controller parameters differ')
                    require(all(actual[k] == v for k, v in fitted['parameters'].items()),
                            'Actual request did not receive fitted parameter')
                    require(args.layer == plan['decoder_output_layer'] == 20
                            and ns['calibration_source_layer'] == 21,
                            'Actual calibration placement changed')
                    expected_sampled = bool(plan['arms'][arm].get('sampled_confidence'))
                    require(bool(actual.get('sampled_confidence')) == expected_sampled
                            and 'negative_only' not in actual,
                            'Candidate parameters mixed')
                    require(actual['think_start_token_id'] == 151648
                            and actual['think_end_token_id'] == 151649
                            and len(actual['boundary_token_ids']) > 0,
                            'Actual boundary/think tokens differ')
                    text = eval(compile(ast.Expression(definition), str(EVAL), 'eval'), ns)
                    require(('actually sampled' in text) == expected_sampled,
                            'Output confidence definition contradicts candidate')
                    require(args.limit == plan['count'] and
                            all(getattr(args, k) == v for k, v in plan['runtime'].items()),
                            'Actual CLI runtime differs from plan')
                    records.append(dict(candidate=candidate, stage=stage, arm=arm,
                        count=args.limit, layer=args.layer, temperature=args.temperature,
                        max_tokens=args.max_tokens, dynamic_params=actual,
                        confidence_definition=text, fit_sha256=sha(args.calibration_fit)))
    finally:
        sys.argv = old_argv
        if previous is missing:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous
    require(len(records) == 12, 'Expected exactly twelve prepared commands')
    return dict(status='actual_12_CLI_fit_and_parameter_branches_passed_CPU_only',
        parent_plan_sha256=sha(bundle / 'plan.json'), tokenizer_sha256=TOKENIZER_SHA,
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p, source=True)
                       for p in (EVAL, CONTRACT, Path(__file__).resolve())},
        records=records,
        limitation='AST subset and frozen-tokenizer CPU audit only. No Torch payload, '
                   'VectorSpec, model forward, probability replay or generation. '
                   'Native preflight remains required before the fixed GPU batch.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit receipt')
    receipt = audit(args.bundle.resolve()); save(args.output, receipt)
    print(json.dumps(dict(status=receipt['status'], commands=len(receipt['records']))))


if __name__ == '__main__':
    main()
