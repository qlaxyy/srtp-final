"""One old/new boundary comparison on <=20 examples; no repeated full runs."""

import gc
import os
import subprocess
import sys
import types
from pathlib import Path

import rebalance_dynamic_eval as evaluation
import torch
from easysteer.vectors import from_pt_direction
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
from vllm.v1.worker.gpu.steer_vector_utils import SteerVectorState
from vllm.v1.worker.gpu import steer_vector_utils as runtime

from rebalance_diagnostics import digest


def main():
    args = evaluation.parse_args()
    if not 1 <= args.limit <= 20:
        raise ValueError('Only 1..20 examples are allowed')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[3]
    dataset = evaluation.resolve_file(args.dataset, 'test.jsonl')
    vector = evaluation.resolve_file(args.vector, 'steer_vector_layer19_conf_mixed.pt')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    boundaries = sorted(i for token, i in tokenizer.get_vocab().items() if 'ĊĊ' in token)
    hp = {key: getattr(args, key) for key in (
        'initial_coef', 'q25c', 'q75c', 'low_val_1', 'q25v', 'q75v', 'low_val_2', 'high_val_2')}
    hp.update(boundary_token_ids=boundaries,
              think_start_token_id=evaluation.single_token_id(tokenizer, '<think>'),
              think_end_token_id=evaluation.single_token_id(tokenizer, '</think>'))
    steering = SteeringSpec(vectors=[VectorSpec(
        name='rebalance_dynamic', data=from_pt_direction(str(vector), layers=[args.layer]),
        algorithm='rebalance', scale=1.0, layers=[args.layer], normalize=False,
        apply=ApplySpec(generation_tokens=boundaries), params=hp)])
    examples = evaluation.load_examples(dataset, args.offset, args.limit)
    prompts = [evaluation.build_prompt(tokenizer, row['problem']) for row in examples]
    result = dict(scope='one-pass boundary implementation comparison', protocol=vars(args),
                  dynamic_params=hp, provenance=dict(
                      commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
                      dataset_sha256=digest(dataset), vector_sha256=digest(vector),
                      model_config_sha256=digest(Path(args.model) / 'config.json'),
                      torch=torch.__version__),
                  tests=[])
    optimized = SteerVectorState._is_boundary
    old_label, new_label = 'T2_old_isin', 'T3_direct_compare'
    before = torch.isin
    def select(function):
        SteerVectorState._is_boundary = staticmethod(function)
    if os.environ.get('REBALANCE_COMPARE_CONSTANTS') == '1':
        # Load the exact previously validated implementation, not a rewritten approximation.
        reference_commit = '713b34e'
        source = subprocess.check_output([
            'git', 'show', reference_commit + ':sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
        ], cwd=root, text=True)
        reference = types.ModuleType('rebalance_constants_reference')
        sys.modules[reference.__name__] = reference
        exec(compile(source, reference_commit + '/rebalance.py', 'exec'), reference.__dict__)
        before = reference.compute_rebalance_coefficient
        optimized = runtime.compute_rebalance_coefficient
        def select(function):
            runtime.compute_rebalance_coefficient = function
        old_label, new_label = 'before_constant_cache', 'after_constant_cache'
        result['scope'] = 'one-pass constant reuse comparison'
        result['provenance']['reference_commit'] = reference_commit
    llm = None
    try:
        llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_steer_vector=True, steer_algorithms=['rebalance'], enforce_eager=False,
                  steer_graph_mode='in_graph', enable_chunked_prefill=False,
                  enable_prefix_caching=False, seed=args.seed)
        sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                  max_tokens=args.max_tokens, seed=args.seed,
                                  skip_special_tokens=True)
        # A single short engineering prompt per implementation, not a full-set warmup.
        for label, function in [('W1', before), ('W2', optimized)]:
            print(label, 'warmup: one engineering prompt, max_tokens=64', flush=True)
            select(function)
            llm.generate([evaluation.build_prompt(tokenizer, 'Compute 2 + 3.')],
                         SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                        max_tokens=64, seed=args.seed),
                         steering=steering, use_tqdm=False)
            result['tests'].append(dict(id=label, kind='warmup', prompts=1, max_tokens=64))
        for label, function in [(old_label, before), (new_label, optimized)]:
            print(label, 'one pass on', len(examples), 'examples', flush=True)
            select(function)
            records, seconds = evaluation.generate_records(
                llm, prompts, examples, sampling, set(boundaries), steering)
            result[label] = dict(summary=evaluation.summarize(records, seconds, args.max_tokens),
                                 records=records)
            result['tests'].append(dict(id=label, kind='evaluation', prompts=len(examples), repeats=1))
            evaluation.write_result(output, result)
        old, new = result[old_label], result[new_label]
        result['token_ids_equal'] = all(a['token_ids'] == b['token_ids'] for a, b in
                                       zip(old['records'], new['records'], strict=True))
        result['time_change_percent'] = (new['summary']['generation_seconds'] /
                                         old['summary']['generation_seconds'] - 1) * 100
        evaluation.write_result(output, result)
        for label in (old_label, new_label):
            print(label, result[label]['summary'], flush=True)
        print('token_ids_equal:', result['token_ids_equal'], flush=True)
    finally:
        select(optimized)
        del llm
        gc.collect()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
