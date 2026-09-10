"""One saved boundary per training question: original displacement versus skip.

The saved prefix is unsteered in both arms. Subsequent boundaries use the frozen
controller. This is a diagnostic, not an independent benchmark or grid search.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def run(args):
    import torch
    import vllm
    from transformers import AutoTokenizer
    from easysteer.vectors import from_pt_direction
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    from vllm.v1.worker.gpu import model_runner as runner_module
    from runtime_guards import guard_dynamic_preemption

    manifest = read(args.prepared/'manifest.json')
    inputs = read(args.prepared/'model_inputs.json')
    assert sha(args.prepared/'model_inputs.json') == manifest['artifacts']['model_inputs_sha256']
    assert len(inputs) == manifest['question_count'] == 20
    assert manifest['temperature'] == 0 and manifest['generated_token_budget_including_prefix'] == 16000
    assert sha(args.vector) == manifest['artifacts']['vector_sha256']
    assert sha(ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl') == manifest['artifacts']['train_sha256']
    model = Path(manifest['model'])
    frozen = read(ROOT/'integration/rebalance_easysteer/configs/final_results_20260909.json')
    verified_model = {}
    for filename, asset in frozen['assets']['models']['1.5B'].items():
        actual = sha(model/filename)
        assert actual == asset['sha256'], filename
        verified_model[filename] = actual
    assert sha(model/'tokenizer.json') == manifest['artifacts']['tokenizer_sha256']
    for rel in ('sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py',):
        source = (ROOT/rel).read_text(encoding='utf-8')
        assert hashlib.sha256(source.encode()).hexdigest() == manifest['artifacts']['controller_git_content_sha256']
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
        raise RuntimeError('Commit the experiment before generating')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    layer = manifest['decoder_output_layer']
    hp = manifest['parameters']
    payload = from_pt_direction(str(args.vector), layers=[layer])
    jobs = []
    for case, data in zip(manifest['cases'], inputs, strict=True):
        assert case['case_id'] == data['case_id']
        assert ids_hash(data['prompt_token_ids']) == case['prompt_sha256']
        assert ids_hash(data['generated_prefix_token_ids']) == case['prefix_sha256']
        assert len(data['generated_prefix_token_ids']) + case['remaining_tokens'] == 16000
        for apply in (True, False):
            jobs.append(dict(case=case, data=data, arm='apply' if apply else 'skip', apply=apply))
    random.Random(20260910).shuffle(jobs)
    prompts = [dict(prompt_token_ids=j['data']['prompt_token_ids']) for j in jobs]
    samplings = [SamplingParams(temperature=0, top_p=.95, seed=42,
        max_tokens=j['case']['remaining_tokens'], skip_special_tokens=True) for j in jobs]
    specs = [SteeringSpec(vectors=[VectorSpec(name=j['case']['case_id']+'_'+j['arm'],
        data=payload, algorithm='rebalance', scale=1., layers=[layer], normalize=False,
        apply=ApplySpec(prompt_positions=[-1], generation_tokens=hp['boundary_token_ids']),
        params=dict(**hp, prefix_mean=j['case']['confidence_float32'],
            prefix_variance=j['case']['variance_float32'], prefix_apply=j['apply']))]) for j in jobs]
    protocol = dict(scope='20 saved training prefixes, one-boundary apply/skip, 40 continuations',
        manifest_sha256=sha(args.prepared/'manifest.json'), cases=manifest['cases'],
        generation_temperature=0, generation_seed=42, total_generated_budget=16000,
        layer=layer, vector_sha256=sha(args.vector), model_files_sha256=verified_model,
        parameters=hp, runtime=dict(dtype='bfloat16', max_model_len=32768, max_num_seqs=40,
            max_num_batched_tokens=32768, gpu_memory_utilization=.90, async_scheduling=True,
            chunked_prefill=False, prefix_caching=False, max_steer_vectors=40),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        torch=torch.__version__, vllm=vllm.__version__, gpu=torch.cuda.get_device_name(0),
        run_order=[dict(case_id=j['case']['case_id'], arm=j['arm']) for j in jobs],
        timing='Shared concurrent wall time includes prefix prefill and new decoding, excludes startup/grading. Per-arm latencies are not isolated speed measurements.')
    write(args.output/'protocol.json', protocol)
    started = time.perf_counter()
    llm = LLM(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=32768, max_num_seqs=40, max_num_batched_tokens=32768,
        gpu_memory_utilization=.90, enable_steer_vector=True, steer_algorithms=['rebalance'],
        max_steer_vectors=40, enforce_eager=False, steer_graph_mode='in_graph',
        enable_chunked_prefill=False, enable_prefix_caching=False, async_scheduling=True, seed=42)
    startup = time.perf_counter()-started
    core = llm.llm_engine.engine_core.engine_core
    state = core.model_executor.driver_worker.worker.model_runner.steer_vector_state
    assert state.supports_kv_replay
    preemptions = guard_dynamic_preemption(core.scheduler, state)
    checks, boundary_seen, req_jobs = [], set(), {}
    original_fill = runner_module.fill_graph_steer_buffers
    prefill_host_seconds = 0.

    def checked_fill(batch, live_state, manager):
        # Validate the actual persistent kernel mask during the formal run;
        # no extra generation/smoke questions are needed.
        before = time.perf_counter()
        original_fill(batch, live_state, manager)
        entries = manager.graph_batch_entries()
        for b, rid in enumerate(batch.req_ids):
            if rid not in req_jobs:
                raise RuntimeError(f'Unknown worker request ID during mask audit: {rid}')
            if rid in boundary_seen:
                continue
            job = req_jobs[rid]
            n = len(job['data']['prompt_token_ids'])
            computed = int(batch.num_computed_tokens_np[b])
            count = int(batch.num_scheduled_tokens[b])
            prefix_count = min(count, n-computed)
            if prefix_count <= 0:
                continue
            idx = live_state._dynamic_indices[rid]
            coef = float(live_state._coefs[idx].item())
            expected_coef = job['case']['projected_coefficient_float32']
            assert abs(coef-expected_coef) < 2e-4, (rid, coef, expected_coef)
            assert float(live_state._prev_step_mean[idx]) == job['case']['confidence_float32']
            assert int(live_state._step_tok_count[idx]) == 0
            slot = live_state._slots[rid]
            _, _, controllers = entries[slot]
            assert controllers
            a = int(batch.query_start_loc_np[b])
            expected = torch.zeros(prefix_count, dtype=torch.float32, device=state._coefs.device)
            includes_boundary = computed + prefix_count == n
            if includes_boundary and job['apply']:
                expected[-1] = coef
            for controller in controllers:
                # Persistent masks use the model dtype (BF16). Compare against
                # the same cast, without loosening the allowed position error.
                expected_mask = expected.to(controller.graph_mask.dtype)
                torch.testing.assert_close(controller.graph_mask[a:a+prefix_count], expected_mask, atol=0, rtol=0)
            if includes_boundary:
                boundary_seen.add(rid)
                checks.append(dict(case_id=job['case']['case_id'], arm=job['arm'],
                    computed_coefficient=coef, applied_coefficient=float(expected_mask[-1]),
                    kernel_mask_dtype=str(controller.graph_mask.dtype),
                    previous_mean=float(live_state._prev_step_mean[idx]),
                    prompt_tokens=n, kernel_prefix_mask_checked=True))
        # Host mask audit cost only, not a claim of total GPU prefill time.
        nonlocal prefill_host_seconds
        prefill_host_seconds += time.perf_counter()-before

    runner_module.fill_graph_steer_buffers = checked_fill
    results = []
    generation_started = time.perf_counter()
    status = 'incomplete'
    try:
        ids = llm.enqueue(prompts, sampling_params=samplings, steering=specs, use_tqdm=False)
        processors = llm.llm_engine.output_processor.request_states
        output_jobs = {processors[rid].external_req_id: jobs[i] for i,rid in enumerate(ids)}
        # Engine and worker use internal IDs; returned outputs use external IDs.
        req_jobs.update({rid:jobs[i] for i,rid in enumerate(ids)})
        with (args.output/'continuations.jsonl').open('x', encoding='utf-8') as stream:
            while llm.llm_engine.has_unfinished_requests():
                batch = llm.llm_engine.step()
                for result in batch:
                    if not result.finished:
                        continue
                    job = output_jobs[result.request_id]
                    assert list(result.prompt_token_ids) == job['data']['prompt_token_ids']
                    assert len(result.outputs) == 1
                    answer = result.outputs[0]
                    new_ids = list(answer.token_ids)
                    all_ids = job['data']['generated_prefix_token_ids'] + new_ids
                    assert len(all_ids) <= 16000 and answer.finish_reason in ('stop','length')
                    record = dict(case_id=job['case']['case_id'], arm=job['arm'],
                        stratum=job['case']['stratum'], train_index=job['case']['train_index'],
                        prefix_generated_tokens=job['case']['prefix_generated_tokens'],
                        new_tokens=len(new_ids), total_tokens=len(all_ids),
                        think_tokens=all_ids.index(hp['think_end_token_id']) if hp['think_end_token_id'] in all_ids else len(all_ids),
                        think_closed=hp['think_end_token_id'] in all_ids,
                        finish_reason=answer.finish_reason, token_ids=new_ids,
                        text=tokenizer.decode(all_ids, skip_special_tokens=True),
                        continuation_text=answer.text,
                        completion_elapsed_seconds=time.perf_counter()-generation_started)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush()
                    results.append(record)
                    print(f"Saved {len(results)}/40: {record['case_id']} {record['arm']} {len(new_ids)} new tokens", flush=True)
            assert len(results) == 40 and len(boundary_seen) == 40
            status = 'completed_generation_ungraded'
    finally:
        torch.cuda.synchronize()
        seconds = time.perf_counter()-generation_started
        runner_module.fill_graph_steer_buffers = original_fill
        write(args.output/'runtime_checks.json', dict(status=status, startup_seconds=startup,
            generation_seconds_including_prefix_prefill=seconds,
            mask_fill_and_audit_host_seconds=prefill_host_seconds,
            replay_counts=state.replay_counts, preemptions=preemptions,
            completed=len(results), checked_boundary_count=len(boundary_seen), checks=checks))
    print(json.dumps(dict(status=status, completed=len(results), seconds=seconds)), flush=True)


def grade(args):
    sys.path.insert(0, str(ROOT/'sources/ReBalance'))
    from utils.parser import extract_answer, parse_ground_truth
    from utils.grader import check_is_correct
    runtime = read(args.output/'runtime_checks.json')
    assert runtime['status'] == 'completed_generation_ungraded' and runtime['checked_boundary_count'] == 40
    records = [json.loads(line) for line in (args.output/'continuations.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [json.loads(line) for line in (ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl').read_text(encoding='utf-8').splitlines() if line]
    manifest = read(args.prepared/'manifest.json')
    expected = {(c['case_id'], arm):c for c in manifest['cases'] for arm in ('apply','skip')}
    assert len(records) == 40 and {(r['case_id'],r['arm']) for r in records} == set(expected)
    started = time.perf_counter()
    scored = []
    for record in records:
        case = expected[record['case_id'],record['arm']]
        assert record['train_index'] == case['train_index']
        _, gold = parse_ground_truth(train[case['train_index']], 'math')
        answer = extract_answer(record['text'], 'math')
        correct = bool(check_is_correct(answer,gold))
        scored.append({k:v for k,v in record.items() if k not in ('text','continuation_text','token_ids')} |
            dict(author_correct=correct, extracted_answer=answer))
    groups = {}
    for stratum in ('strong_negative','positive'):
        rows = [r for r in scored if r['stratum']==stratum]
        by_key = {(r['case_id'],r['arm']):r for r in rows}
        pair_rows = []
        for cid in sorted({r['case_id'] for r in rows}):
            a,b = by_key[cid,'apply'],by_key[cid,'skip']
            outcome = ('both_correct' if a['author_correct'] and b['author_correct'] else
                       'apply_only_correct' if a['author_correct'] else
                       'skip_only_correct' if b['author_correct'] else 'both_wrong')
            pair_rows.append(dict(case_id=cid,outcome=outcome,apply_minus_skip_tokens=a['total_tokens']-b['total_tokens']))
        arms = {}
        for arm in ('apply','skip'):
            selected = [r for r in rows if r['arm']==arm]
            arms[arm] = dict(count=len(selected), correct=sum(r['author_correct'] for r in selected),
                mean_total_tokens=sum(r['total_tokens'] for r in selected)/len(selected),
                mean_new_tokens=sum(r['new_tokens'] for r in selected)/len(selected),
                mean_think_tokens=sum(r['think_tokens'] for r in selected)/len(selected),
                capped=sum(r['finish_reason']=='length' for r in selected),
                thinking_not_closed=sum(not r['think_closed'] for r in selected))
        groups[stratum] = dict(arms=arms,pairs=pair_rows,
            outcomes={s:sum(r['outcome']==s for r in pair_rows) for s in
                ('both_correct','apply_only_correct','skip_only_correct','both_wrong')})
    summary = dict(status='completed', scope='20 calibration training prefixes, single-boundary action diagnostic',
        question_count=20, continuations=40, grading='Released author math grader on full prefix plus continuation; errors/caps retained',
        grading_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        grader_sha256={name:sha(ROOT/'sources/ReBalance/utils'/name) for name in ('parser.py','grader.py')},
        groups=groups, records=scored, runtime=runtime,
        grading_seconds=time.perf_counter()-started,
        hashes={name:sha(args.output/name) for name in ('protocol.json','continuations.jsonl','runtime_checks.json')},
        limitations=['Training calibration data, not an independent validation or test set.',
            'Two coefficient strata sampled deliberately; aggregate rates are not population rates.',
            'One greedy continuation per action and boundary; no claim of general causal benefit.',
            'Saved prefix statistics use float32 CPU accumulation, not an exact CUDA probability replay.',
            'All requests share concurrent scheduling; timing is diagnostic, not a speed benchmark.'])
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(groups=groups, runtime_seconds=runtime['generation_seconds_including_prefix_prefill']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','grade'])
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vector',type=Path,default=Path('/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908/auto_vector.pt'))
    args = parser.parse_args()
    (run if args.mode=='run' else grade)(args)
