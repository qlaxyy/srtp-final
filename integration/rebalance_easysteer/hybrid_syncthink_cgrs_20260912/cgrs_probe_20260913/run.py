"""Bounded C-probe engineering/screen runner. Never opens SSH or installs packages."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from policy import Config, State, TRIGGERS, PROBE_PROMPT, boxed_certainty, problem_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NS = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def source_sha(path):
    """Canonical LF source hash, stable across Windows checkout/git archive."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def deployment_record(root=ROOT):
    """A hash-checked archive deployment need not contain a Git checkout."""
    import subprocess
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                            capture_output=True, text=True)
    manifest = root/'cgrs_deployment_identity.json'
    return dict(git_commit=result.stdout.strip() if result.returncode == 0 else None,
                git_metadata_available=result.returncode == 0,
                archive_deployment=read(manifest) if manifest.exists() else None)


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def validate(plan):
    if plan['phase'] not in ('engineering', 'screen'):
        raise ValueError('Only engineering or fresh training screening is implemented')
    if not plan['coordination']['data_reconciled'] or not plan['coordination']['gpu_authorized']:
        raise ValueError('Executor must reconcile data and record batch authorization')
    if (not plan['run_id'].startswith('cgrs_probe_')
            or '/' in plan['run_id'] or '\\' in plan['run_id']):
        raise ValueError('Use independent C-probe run_id')
    expected = 8 if plan['phase'] == 'engineering' else 64
    rows = plan['rows']
    if len(rows) != expected or plan['dataset'] not in ('math', 'gsm8k'):
        raise ValueError('Fixed scope is 8 engineering or 64 screening questions')
    for row in rows:
        if (row['problem_sha256'] != problem_hash(row['problem'])
                or row['split'] != 'train' or 'train_index' not in row):
            raise ValueError('Invalid training identity')
    if len({r['problem_sha256'] for r in rows}) != expected:
        raise ValueError('Duplicate question')
    if len({r['train_index'] for r in rows}) != expected:
        raise ValueError('Duplicate source index')
    if plan['source_sha256'] != read(HERE/'identity.json')['source_sha256']:
        raise ValueError('Plan and implementation identity differ')
    for name, expected_hash in plan['source_sha256'].items():
        if source_sha(ROOT/name) != expected_hash:
            raise ValueError(f'Changed source: {name}')
    assets = plan['assets']
    if assets != read(HERE/'identity.json')['assets']:
        raise ValueError('This implementation is pinned to the existing 1.5B assets')
    for name, item in assets['model_files'].items():
        if sha(Path(assets['model_path'])/name) != item['sha256']:
            raise ValueError(f'Changed model asset: {name}')
    for key in ('vector', 'fit'):
        if sha(assets[key]['path']) != assets[key]['sha256']:
            raise ValueError(f'Changed {key}')
    if plan['phase'] == 'screen':
        gate = read(plan['engineering_gate']['path'])
        if (sha(plan['engineering_gate']['path']) != plan['engineering_gate']['sha256']
                or not gate['passed'] or gate['source_sha256'] != plan['source_sha256']
                or gate['assets'] != assets or not gate['replay_checks']
                or not all(x['passed'] for x in gate['replay_checks'])):
            raise ValueError('Missing matching successful native engineering gate')
    return rows


def build_engine(plan):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['PYTHONNOUSERSITE'] = '1'
    sys.path[:0] = [str(ROOT/'sources/EasySteer/vllm-steer'),
                   str(ROOT/'sources/EasySteer'),
                   str(ROOT/'integration/rebalance_easysteer/eval')]
    from transformers import AutoTokenizer
    from vllm import LLM
    from easysteer.vectors import from_pt_direction
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    assets = plan['assets']
    tok = AutoTokenizer.from_pretrained(assets['model_path'], local_files_only=True)
    for token, text in TRIGGERS.items():
        if tok.decode([token]) != text or tok.encode(text, add_special_tokens=False) != [token]:
            raise ValueError(f'Trigger tokenizer mismatch: {token}')
    boundaries = sorted(i for t, i in tok.get_vocab().items() if 'ĊĊ' in t)
    fit = read(assets['fit']['path'])
    steering = SteeringSpec(vectors=[VectorSpec(
        name='rebalance_dynamic',
        data=from_pt_direction(assets['vector']['path'], layers=[20]),
        algorithm='rebalance', scale=1., layers=[20], normalize=False,
        apply=ApplySpec(generation_tokens=boundaries),
        params=dict(fit['parameters'], boundary_token_ids=boundaries,
                    think_start_token_id=151648, think_end_token_id=151649))])
    llm = LLM(model=assets['model_path'], dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=32768, max_num_seqs=256, max_num_batched_tokens=32768,
              gpu_memory_utilization=.9, enable_steer_vector=True,
              steer_algorithms=['rebalance'], steer_graph_mode='in_graph',
              enforce_eager=False, enable_chunked_prefill=False,
              enable_prefix_caching=False, async_scheduling=False, seed=42)
    from backend import Backend
    llm.llm_engine.engine_core.engine_core.scheduler._preempt_request = Backend.reject_preemption
    return llm, tok, steering, boundaries


def run_arm(llm, tok, steering, boundaries, rows, arm, plan, output):
    import torch
    from vllm import SamplingParams
    from rebalance_static_eval import build_prompt
    from backend import Backend, parked
    folder = output/arm
    folder.mkdir(exist_ok=False)
    engineering = plan['phase'] == 'engineering'
    cfg = Config(interval=32, max_probes=2, probe_tokens=8, max_tokens=256) if engineering else Config()
    c_on = arm in ('C', 'RC', 'Rshadow')
    r_on = arm != 'C'
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    if core.scheduler.requests or llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('Previous arm not drained')
    backend = Backend(llm, suppress=arm in ('C', 'RC')) if arm != 'R' else None
    started = time.monotonic()
    deadline = started + (180 if engineering else 600)
    if backend:
        backend.deadline = deadline
        backend.tokenizer = tok
    sampling = SamplingParams(temperature=.7, top_p=.95, seed=42,
                              max_tokens=cfg.max_tokens, skip_special_tokens=False)
    prompt_ids = [tok.encode(build_prompt(tok, row['problem'])) for row in rows]
    if max(map(len, prompt_ids)) + cfg.max_tokens >= 32768:
        raise ValueError('Prompt exceeds model budget')
    history_hash = {}
    state = runner.steer_vector_state
    original_remove = state.remove_request

    def capture(rid, manager):
        if rid in state._dynamic_indices:
            idx = state._dynamic_indices[rid]
            history_hash[rid] = hashlib.sha256(
                state._history[idx, :state._history_lengths[rid]].cpu().numpy().tobytes()).hexdigest()
        return original_remove(rid, manager)

    state.remove_request = capture
    torch.cuda.synchronize()
    ids = llm.enqueue([{'prompt_token_ids': x} for x in prompt_ids],
                      sampling_params=sampling, steering=steering if r_on else None,
                      use_tqdm=False)
    ops = llm.llm_engine.output_processor.request_states
    external = {ops[rid].external_req_id: rid for rid in ids}
    row_map = dict(zip(ids, rows))
    prompts = dict(zip(ids, prompt_ids))
    policies = {rid: State(row_map[rid]['problem_sha256'], cfg) for rid in ids}
    if backend and c_on:
        backend.states = policies
    latest, done, due, probes = {}, {}, set(), []
    suffix = tok.encode(PROBE_PROMPT, add_special_tokens=False)
    probe_seconds, probe_prefill_tokens, audit_tokens, io_seconds = 0., 0, 0, 0.
    preservation = []
    try:
        with (folder/'partial.jsonl').open('x', encoding='utf8') as partial, \
                (folder/'probes.jsonl').open('x', encoding='utf8') as probe_stream:
            while len(done) < len(ids):
                if time.monotonic() > deadline:
                    raise TimeoutError('Per-arm wall limit; retain partials')
                for result in llm.llm_engine.step():
                    rid = external[result.request_id]
                    completion = result.outputs[0]
                    tokens = list(completion.token_ids)
                    policy = policies[rid]
                    if tokens[:policy.count] != latest.get(rid, []):
                        raise RuntimeError('Expected cumulative accepted output')
                    for token in tokens[policy.count:]:
                        trigger = policy.accept(token, tok.decode([token]), boundaries)
                        if trigger and c_on:
                            due.add(rid)
                    latest[rid] = tokens
                    if result.finished:
                        due.discard(rid)
                        think = tokens.index(151649) if 151649 in tokens else len(tokens)
                        rec = dict(row_map[rid], token_ids=tokens,
                                   text=tok.decode(tokens, skip_special_tokens=True),
                                   tokens=len(tokens), thinking_tokens=think,
                                   finish_reason=completion.finish_reason,
                                   policy=asdict(policy), request_id=rid)
                        done[rid] = rec
                        before = time.perf_counter()
                        partial.write(json.dumps(rec, ensure_ascii=False)+'\n')
                        partial.flush()
                        io_seconds += time.perf_counter() - before
                if not due or core.scheduler.waiting or core.scheduler.skipped_waiting:
                    continue
                pending = []
                for rid in ids:
                    if rid not in due or rid in done:
                        continue
                    policy = policies[rid]
                    # If admission delayed probing past the boundary, drop it.
                    if not policy.opening or not policy.thinking:
                        continue
                    if policy.reserve_probe(len(suffix)):
                        prefix = prompts[rid] + latest[rid]
                        pending.append(dict(rid=rid, prefix=prefix,
                                            snapshot=backend.snapshot(rid, len(prefix))))
                due.clear()
                if not pending:
                    continue
                began = time.monotonic()
                with parked(core.scheduler) as held:
                    before = backend.primary_signature(held)
                    if engineering and arm == 'Rshadow':
                        audit_sampling = SamplingParams(temperature=0, max_tokens=1,
                                                        skip_special_tokens=False)
                        audits = backend.drain_probe_requests(
                            pending, audit_sampling, steering if r_on else None, [], audit=True)
                        for item, audit in zip(pending, audits):
                            backend.expected_logits[item['rid']] = audit['logits']
                        audit_tokens += sum(len(a['token_ids']) for a in audits)
                    ps = SamplingParams(temperature=0, max_tokens=cfg.probe_tokens,
                                        skip_special_tokens=False)
                    results = backend.drain_probe_requests(
                        pending, ps, steering if r_on else None, suffix)
                    after = backend.primary_signature(held)
                    if before != after:
                        raise RuntimeError('Probe changed primary tokens or R controller')
                    preservation.append(dict(before=before, after=after, passed=True))
                    for item, result in zip(pending, results):
                        rid = item['rid']
                        certainty, status, selected = boxed_certainty(
                            result['token_ids'], result['entropy'], tok, result['vocab_size'])
                        policy = policies[rid]
                        policy.complete_probe(certainty, len(result['token_ids']))
                        prefill = len(item['prefix']) + len(suffix)
                        probe_prefill_tokens += prefill
                        record = dict(parent_key=policy.key, position=policy.count,
                                      token_ids=result['token_ids'], entropy=result['entropy'],
                                      selected_tokens=selected, certainty=certainty,
                                      status=status, suppression_probability=policy.p,
                                      prefill_tokens=prefill,
                                      prefix_sha256=hashlib.sha256(
                                          json.dumps(item['prefix']).encode()).hexdigest())
                        probes.append(record)
                        start_io = time.perf_counter()
                        probe_stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                        probe_stream.flush()
                        io_seconds += time.perf_counter() - start_io
                        if not engineering:
                            # Scheduler enforces the combined main/probe budget.
                            req = core.scheduler.requests[rid]
                            req.max_tokens = (cfg.max_tokens - policy.probe_output_tokens
                                              - policy.probe_prompt_tokens)
                            req.sampling_params.max_tokens = req.max_tokens
                            ops[rid].max_tokens_param = req.max_tokens
                probe_seconds += time.monotonic() - began
            torch.cuda.synchronize()
            seconds = time.monotonic() - started - io_seconds
            llm.llm_engine.step()
            for rid in ids:
                if rid in runner.req_states.req_id_to_index:
                    runner._remove_request(rid)
            records = []
            for rid in ids:
                rec, policy = done[rid], policies[rid]
                rec['R_history_sha256'] = history_hash.get(rid)
                rec['all_branch_output_tokens'] = rec['tokens'] + policy.probe_output_tokens
                rec['budget_tokens_including_probe_prompt'] = (
                    rec['all_branch_output_tokens'] + policy.probe_prompt_tokens)
                if not engineering and rec['budget_tokens_including_probe_prompt'] > cfg.max_tokens:
                    raise RuntimeError('Combined token budget exceeded')
                records.append(rec)
            summary = dict(status='complete', arm=arm, config=asdict(cfg), records=records,
                           generation_seconds=seconds, probe_wall_seconds=probe_seconds,
                           probe_prefill_tokens=probe_prefill_tokens,
                           probe_output_tokens=sum(p.probe_output_tokens for p in policies.values()),
                           probe_prompt_tokens=sum(p.probe_prompt_tokens for p in policies.values()),
                           audit_output_tokens=audit_tokens, checkpoint_io_seconds=io_seconds,
                           extra_forward_counts=backend.forward_counts if backend else {},
                           callback_host_seconds=backend.callback_host_seconds if backend else 0.,
                           replay_checks=backend.replay_checks if backend else [],
                           primary_preservation_checks=preservation,
                           limitations=['Synchronous pilot; not original async speed benchmark',
                                        'Replay prefill compute included in generation time',
                                        'Engineering shadow does not debit probe budget from main'])
            save(folder/'result.json', summary)
            return summary
    except BaseException as exc:
        save(folder/'partial_state.json', dict(
            error=repr(exc), accepted_main_tokens=latest,
            policies={rid: asdict(p) for rid, p in policies.items()},
            probe_records=probes,
            active_probe_entropy={rid: p['entropy'] for rid, p in backend.probes.items()}
            if backend else {},
            replay_checks=backend.replay_checks if backend else [],
            primary_preservation_checks=preservation))
        raise
    finally:
        state.remove_request = original_remove
        if backend:
            backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    rows = validate(plan)
    output = args.output_root/plan['run_id']
    output.mkdir(parents=True, exist_ok=False)
    save(output/'resolved_plan.json', plan)
    started = time.monotonic()
    try:
        import subprocess
        check = subprocess.run([sys.executable, str(HERE/'test_native.py')],
                               capture_output=True, text=True, timeout=60)
        save(output/'native_cpu_check.json', dict(returncode=check.returncode,
                                                 stdout=check.stdout, stderr=check.stderr))
        if check.returncode:
            raise RuntimeError('Native torch CPU tests failed before model loading')
        llm, tok, steering, boundaries = build_engine(plan)
        startup = time.monotonic() - started
        import torch, vllm
        save(output/'runtime_identity.json', dict(
            python=sys.version, torch=torch.__version__, vllm=vllm.__version__,
            vllm_path=vllm.__file__, plan_sha256=sha(args.plan),
            deployment=deployment_record(),
            source_sha256=plan['source_sha256'], assets=plan['assets']))
        arms = ['R', 'Roff', 'Rshadow', 'C', 'RC'] if plan['phase'] == 'engineering' else ['R', 'C', 'RC']
        results = {}
        for arm in arms:
            results[arm] = run_arm(llm, tok, steering, boundaries, rows, arm, plan, output)
        if plan['phase'] == 'engineering':
            r = results['R']['records']
            equivalent = all(
                [(x['token_ids'], x['R_history_sha256']) for x in results[a]['records']]
                == [(x['token_ids'], x['R_history_sha256']) for x in r]
                for a in ('Roff', 'Rshadow'))
            checks = results['Rshadow']['replay_checks']
            passed = equivalent and bool(checks) and all(x['passed'] for x in checks)
            save(output/'engineering_gate.json', dict(
                passed=passed, off_shadow_equivalent=equivalent, replay_checks=checks,
                source_sha256=plan['source_sha256'], assets=plan['assets'],
                runtime=dict(python=sys.version), config='sync_tp1_bf16_in_graph',
                result_sha256={a: sha(output/a/'result.json') for a in arms}))
        save(output/'batch_status.json', dict(status='complete', startup_seconds=startup,
                                              wall_seconds=time.monotonic()-started))
    except BaseException as exc:
        save(output/'failure.json', dict(status='failed', error=repr(exc),
                                        wall_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
