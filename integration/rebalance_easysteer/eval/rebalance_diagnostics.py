"""Fixed <=20-example diagnostics; instrumentation is separate from timing."""

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import rebalance_dynamic_eval as evaluation
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu import steer_vector_utils as runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "quantiles_0_10_25_50_75_90_100": tensor.quantile(
            torch.tensor([0, .1, .25, .5, .75, .9, 1], dtype=torch.float64)
        ).tolist(),
        "negative": int((tensor < 0).sum()),
        "positive": int((tensor > 0).sum()),
        "near_zero_abs_lt_0.01": int((tensor.abs() < .01).sum()),
    }


class BoundaryTrace:
    """GPU snapshots exported only after generation; records updates, not injection."""

    def __init__(self):
        self.snapshots = []

    def observe(self, original, state, batch, tokens, probabilities):
        indices = batch.idx_mapping[:batch.num_reqs].long()
        before = torch.stack([
            state._step_prob_sum.index_select(0, indices),
            state._step_tok_count.index_select(0, indices).float(),
            state._prev_step_mean.index_select(0, indices),
        ], dim=1)
        original(state, batch, tokens, probabilities)
        after = torch.stack([
            tokens[:, 0].float(),
            state._coefs.index_select(0, indices),
            state._in_think.index_select(0, indices).float(),
        ], dim=1)
        self.snapshots.append((list(batch.req_ids), torch.cat([before, after], 1)))

    def export(self, boundary_ids):
        events = []
        request_order = {}
        offsets = {}
        for req_ids, snapshot in self.snapshots:
            for req_id, row in zip(req_ids, snapshot.cpu().tolist(), strict=True):
                local_index = request_order.setdefault(req_id, len(request_order))
                token_offset = offsets.get(req_id, 0)
                offsets[req_id] = token_offset + 1
                total, count, previous, token, coefficient, in_think = row
                if int(token) not in boundary_ids or count <= 0:
                    continue
                confidence = total / count
                variance = (confidence - previous)**2 / 4 if math.isfinite(previous) else 0
                events.append(dict(
                    request_index=local_index, token_offset=token_offset,
                    confidence=confidence, variance=variance, coefficient=coefficient,
                    in_think=bool(in_think), step_tokens=int(count),
                ))
        active = [event for event in events if event['in_think']]
        return {
            "scope": "boundary coefficient updates; excludes initial coefficient; not an injection count",
            "decode_calls": len(self.snapshots),
            "request_rows": sum(len(ids) for ids, _ in self.snapshots),
            "all_boundary_updates": len(events),
            "in_think_updates": len(active),
            "coefficient": distribution([e['coefficient'] for e in active]),
            "confidence": distribution([e['confidence'] for e in active]),
            "variance": distribution([e['variance'] for e in active]),
            "events": events,
        }


def main():
    args = evaluation.parse_args()
    if not 1 <= args.limit <= 20:
        raise ValueError('Diagnostics are restricted to 1..20 examples')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    root = Path(__file__).resolve().parents[3]
    provenance = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
        'dataset_sha256': digest(evaluation.resolve_file(args.dataset, 'test.jsonl')),
        'vector_sha256': digest(evaluation.resolve_file(args.vector, 'steer_vector_layer19_conf_mixed.pt')),
        'model_config_sha256': digest(Path(args.model) / 'config.json'),
    }
    diagnostics = {'provenance': provenance, 'timing_protocol':
                   '64-token warmup per mode; three uninstrumented repeats; baseline then corrected dynamic; no interleaving'}
    original_generate = evaluation.generate_records

    def run(llm, prompts, examples, sampling, boundary_ids, steering):
        label = 'baseline' if steering is None else 'corrected_dynamic'
        llm.generate(prompts, SamplingParams(temperature=args.temperature,
                     top_p=args.top_p, seed=args.seed, max_tokens=64),
                     steering=steering, use_tqdm=False)
        runs = [original_generate(llm, prompts, examples, sampling, boundary_ids, steering)
                for _ in range(3)]
        reference = [r['token_ids'] for r in runs[0][0]]
        diagnostics[label] = {
            'seconds': [seconds for _, seconds in runs],
            'median_seconds': statistics.median(seconds for _, seconds in runs),
            'repeat_token_ids_equal': all([r['token_ids'] for r in records] == reference for records, _ in runs),
            'repeat_summaries': [evaluation.summarize(records, seconds, args.max_tokens) for records, seconds in runs],
        }
        if steering is not None:
            trace = BoundaryTrace()
            original_observe = runtime.SteerVectorState.observe_sample
            runtime.SteerVectorState.observe_sample = lambda state, batch, tokens, probs: trace.observe(
                original_observe, state, batch, tokens, probs)
            try:
                records, seconds = original_generate(llm, prompts, examples, sampling, boundary_ids, steering)
            finally:
                runtime.SteerVectorState.observe_sample = original_observe
            diagnostics['trace'] = trace.export(boundary_ids)
            diagnostics['trace']['token_ids_equal_uninstrumented'] = [r['token_ids'] for r in records] == reference
            diagnostics['trace']['instrumented_seconds_not_benchmark'] = seconds
            if not trace.snapshots:
                raise RuntimeError('No observations captured: worker hooks are not active')
            del trace

            # Separate bounded profiler pass; never use its elapsed time as throughput.
            original_curve = runtime.compute_rebalance_coefficient
            original_scales = runtime.SteerVectorState.batch_scales
            def curve(*arguments):
                with torch.profiler.record_function('rebalance.controller'):
                    return original_curve(*arguments)
            def scales(*arguments):
                with torch.profiler.record_function('rebalance.batch_scales'):
                    return original_scales(*arguments)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=1, active=32, repeat=1),
            ) as profiler:
                def observe(*arguments):
                    with torch.profiler.record_function('rebalance.observe_sample'):
                        original_observe(*arguments)
                    profiler.step()
                runtime.compute_rebalance_coefficient = curve
                runtime.SteerVectorState.batch_scales = scales
                runtime.SteerVectorState.observe_sample = observe
                try:
                    llm.generate(prompts, SamplingParams(temperature=args.temperature,
                                 top_p=args.top_p, seed=args.seed, max_tokens=64),
                                 steering=steering, use_tqdm=False)
                finally:
                    runtime.compute_rebalance_coefficient = original_curve
                    runtime.SteerVectorState.batch_scales = original_scales
                    runtime.SteerVectorState.observe_sample = original_observe
            profiler.export_chrome_trace(str(output.with_suffix('.trace.json')))
            diagnostics['profile'] = [dict(
                name=event.key, calls=event.count,
                cpu_total_us=event.cpu_time_total, self_cpu_us=event.self_cpu_time_total,
                device_total_us=event.device_time_total, self_device_us=event.self_device_time_total,
            ) for event in profiler.key_averages()]
        # Main result's timing belongs to its saved first-run records.
        return runs[0]

    evaluation.generate_records = run
    try:
        evaluation.main()
    finally:
        evaluation.generate_records = original_generate
        if output.exists():
            result = json.loads(output.read_text())
            result['diagnostics'] = diagnostics
            evaluation.write_result(output, result)
    print(json.dumps({k: v for k, v in diagnostics.items() if k not in ('trace', 'profile')}, indent=2))
    print('Trace summary:', {k: v for k, v in diagnostics.get('trace', {}).items() if k != 'events'})


if __name__ == '__main__':
    main()
