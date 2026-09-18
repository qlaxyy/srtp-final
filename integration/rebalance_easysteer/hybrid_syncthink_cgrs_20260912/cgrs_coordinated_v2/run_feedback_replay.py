"""Replay 16 complete saved responses through native L27, twice; no new answers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((a.input/'plan.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed 900s process ceiling; retain partials')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan.get('process_ceiling_seconds',900))
    try:
        for n, h in plan['input_sha256'].items():
            assert sha(a.input/n) == h, n
        runtime = Path(plan['runtime_root'])
        for n, h in plan['runtime_source_sha256'].items():
            assert sha(runtime/n) == h, n
        assets = plan['assets']
        for n, item in assets['model_files'].items():
            assert sha(Path(assets['model_path'])/n) == item['sha256'], n
        for k in ['vector', 'fit']:
            assert sha(assets[k]['path']) == assets[k]['sha256'], k
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', PYTHONNOUSERSITE='1')
        sys.path[1:1] = [str(runtime/'sources/EasySteer/vllm-steer'), str(runtime/'sources/EasySteer')]
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.steer_vectors import SteeringSpec, VectorSpec, ApplySpec
        from easysteer.vectors import from_pt_direction
        from label_alignment_adapter import AlignmentAdapter
        from feedback_replay_capture import FeedbackCapture
        from motivation_replay_sampler import ReplayBuffers, MotivationReplaySampler
        tok = AutoTokenizer.from_pretrained(assets['model_path'], local_files_only=True)
        boundaries = sorted(i for s, i in tok.get_vocab().items() if 'ĊĊ' in s)
        rows = json.loads((a.input/'rows.json').read_text())
        llm = LLM(model=assets['model_path'], dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=16384, max_num_seqs=plan['runtime']['max_num_seqs'], max_num_batched_tokens=32768,
            gpu_memory_utilization=.9, enable_steer_vector=True, steer_algorithms=['rebalance'],
            steer_graph_mode='in_graph', enforce_eager=False, enable_chunked_prefill=False,
            enable_prefix_caching=False, async_scheduling=True, seed=42)
        core = llm.llm_engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        fit = json.loads(Path(assets['fit']['path']).read_text())
        layer = assets['decoder_output_layer']
        steer = SteeringSpec(vectors=[VectorSpec(name='feedback_frozen_L27',
            data=from_pt_direction(assets['vector']['path'], layers=[layer]), algorithm='rebalance',
            scale=1., layers=[layer], normalize=False, apply=ApplySpec(generation_tokens=boundaries),
            params=dict(fit['parameters'], boundary_token_ids=boundaries, think_start_token_id=151648, think_end_token_id=151649))])
        tables = dict(np.load(a.input/'opening.npz'))
        cap = max(len(r['token_ids']) for r in rows)
        summaries = []
        histories = {}
        modes = ['scoring_capture'] if plan.get('capture_only') else ['legacy_forcer', 'scoring_capture']
        if plan.get('legacy_reference'):
            ref = plan['legacy_reference']
            directory = Path(ref['directory'])
            assert sha(directory/'legacy_forcer_manifest.json') == ref['manifest_sha256']
            manifest = json.loads((directory/'legacy_forcer_manifest.json').read_text())
            assert {r['key'] for r in manifest} == {r['key'] for r in rows}
            for item in manifest:
                assert sha(directory/item['file']) == item['sha256']
                histories[item['key']] = np.load(directory/item['file'])['history']
            modes = ['scoring_capture']
            summaries.append(dict(mode='legacy_forcer_reused', **ref))
        for mode in modes:
            adapter = AlignmentAdapter(llm, tok, tables=tables, large_suppression=True, enabled=True)
            capture = FeedbackCapture(adapter, cap) if mode == 'scoring_capture' else None
            buffers = ReplayBuffers(runner.max_num_reqs, cap, runner.device) if capture is None else None
            if capture:
                capture.install()
            else:
                adapter.original_sampler = MotivationReplaySampler(adapter.original_sampler, buffers)
            original_add, original_remove = runner.add_requests, runner._remove_request
            original_observe = runner.steer_vector_state.observe_sample
            pending, active, completed = {}, {}, {}
            def add(output):
                original_add(output)
                for r in output.scheduled_new_reqs:
                    slot = runner.req_states.req_id_to_index[r.req_id]
                    row = pending[r.req_id]
                    active[r.req_id] = slot
                    if capture:
                        capture.register(slot, row['token_ids'])
                    else:
                        buffers.register(slot, row['token_ids'], len(row['prompt_token_ids']))
            def observe(batch, tokens, probabilities, *rest):
                if capture:
                    valid = torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens >= batch.prefill_len_np, device=runner.device)
                    torch._assert_async((~valid | (probabilities == capture.last_raw)).all(), 'Raw confidence changed')
                return original_observe(batch, tokens, probabilities, *rest)
            def remove(rid):
                if rid in active:
                    slot = active.pop(rid)
                    r = pending[rid]
                    n = len(r['token_ids'])
                    state = runner.steer_vector_state
                    history = state._history[slot, :len(r['prompt_token_ids'])+n].cpu().numpy().copy()
                    assert np.array_equal(history[:len(r['prompt_token_ids'])], np.zeros(len(r['prompt_token_ids'])))
                    assert history[-1] == 0, 'EOS after thinking must not be steered'
                    data = dict(history=history)
                    if capture:
                        data.update(capture.completed(slot))
                    else:
                        buffers.completed(slot)
                    key = r['key']
                    if capture and not plan.get('capture_only'):
                        assert np.array_equal(history, histories[key]), (key, 'instrumentation changed native control history')
                    else:
                        histories[key] = history
                    file = mode+'_'+key+'.npz'
                    np.savez_compressed(a.output/file, **data)
                    completed[rid] = dict(key=key, file=file, sha256=sha(a.output/file), response_tokens=n,
                        history_sha256=hashlib.sha256(history.tobytes()).hexdigest())
                return original_remove(rid)
            runner.add_requests, runner._remove_request = add, remove
            runner.steer_vector_state.observe_sample = observe
            params = [SamplingParams(temperature=0, seed=42, max_tokens=len(r['token_ids']), ignore_eos=True, skip_special_tokens=False) for r in rows]
            t0 = time.monotonic()
            ids = llm.enqueue([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows], sampling_params=params, steering=steer, use_tqdm=False)
            pending.update(zip(ids, rows))
            states = llm.llm_engine.output_processor.request_states
            mapping = {states[r].external_req_id:r for r in ids}
            answers = {}
            with (a.output/(mode+'_partial.jsonl')).open('x') as fp:
                while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    for output in llm.llm_engine.step():
                        assert output.finished
                        rid = mapping[output.request_id]
                        tokens = list(output.outputs[0].token_ids)
                        assert tokens == pending[rid]['token_ids'], 'Forced target mismatch'
                        answers[rid] = tokens
                        fp.write(json.dumps(dict(key=pending[rid]['key'], token_count=len(tokens), status='saved_tokens_complete'))+'\n')
                        fp.flush()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:
                        runner._remove_request(rid)
            torch.cuda.synchronize()
            assert len(answers) == len(rows) and len(completed) == len(rows) and not active
            save(a.output/(mode+'_manifest.json'), [completed[r] for r in ids])
            summaries.append(dict(mode=mode, responses=len(rows), seconds=time.monotonic()-t0, saved_tokens=sum(len(r['token_ids']) for r in rows)))
            save(a.output/'progress.json', summaries)
            runner.add_requests, runner._remove_request = original_add, original_remove
            runner.steer_vector_state.observe_sample = original_observe
            if capture:
                capture.close()
            else:
                adapter.original_sampler = adapter.original_sampler.original
            adapter.close()
        save(a.output/'complete.json', dict(status='Native L27 forced-scoring trace complete; no fitting or efficacy claim',
            groups=summaries, history_exact=None if plan.get('capture_only') else True, native_history_recorded=True, raw_confidence_exact=True, wall_seconds=time.monotonic()-start,
            new_generation_count=0, plan_sha256=sha(a.input/'plan.json')))
    except BaseException:
        save(a.output/'failure.json', dict(error=traceback.format_exc(), wall_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
