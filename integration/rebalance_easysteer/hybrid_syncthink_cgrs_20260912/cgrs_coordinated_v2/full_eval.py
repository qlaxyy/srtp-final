"""Fixed full-test coordinated RC only; frozen U/R are read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

from engineering import ROOT, HERE, read, save, sha

ROLES = ['math_test', 'gsm8k_test']


def validate(plan, receipt, plan_path):
    assert plan['phase']=='full_test' and plan['arms']==['RCnegative']
    assert list(plan['datasets'])==ROLES
    assert [len(plan['datasets'][r]['rows']) for r in ROLES]==[500,1319]
    assert plan['runtime']['max_tokens']==16000
    assert receipt['plan_sha256']==sha(plan_path) and receipt['gpu_authorized'] is True
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name, True) == digest, name
    a = plan['assets']
    for name, meta in a['model_files'].items():
        assert sha(Path(a['model_path'])/name) == meta['sha256'], name
    for key in ('vector', 'fit'):
        assert sha(a[key]['path']) == a[key]['sha256'], key
    hashes = set()
    for row in [r for sub in plan['datasets'].values() for r in sub['rows']]:
        text = ''.join(unicodedata.normalize('NFKC', row['problem']).split())
        h = hashlib.sha256(text.encode()).hexdigest()
        assert row['split'] == 'test' and h == row['problem_sha256'] and h not in hashes
        hashes.add(h)
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()


def historical(plan):
    history = {}
    for role, sub in plan['datasets'].items():
        meta = plan['frozen_benchmarks'][role]['artifacts']
        folder = Path('/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908')
        ep, gp = folder/meta['evaluation'], folder/meta['grading']
        assert sha(ep) == meta['evaluation_sha256']
        assert sha(gp) == meta['grading_sha256']
        evaluation, grades = read(ep), read(gp)
        assert grades['input_sha256'] == sha(ep)
        for name in ('parser.py', 'grader.py'):
            assert sha(ROOT/'sources/ReBalance/utils'/name) == grades[name+'_sha256']
        for key, value in dict(max_tokens=16000, max_model_len=32768,
                               temperature=.7, top_p=.95, seed=42, easysteer_output_layer=20).items():
            assert evaluation['protocol'][key] == value
        assert evaluation['provenance']['vector_sha256'] == sub['assets']['vector']['sha256']
        assert evaluation['provenance']['calibration_fit_sha256'] == sub['assets']['fit']['sha256']
        groups = {}
        for arm, key in [('U', 'baseline'), ('R', 'rebalance_dynamic')]:
            records, labels = evaluation[key]['records'], grades['groups'][key]['records']
            assert len(records) == len(labels) == len(sub['rows'])
            compact = []
            for i, (x, g, row) in enumerate(zip(records, labels, sub['rows'])):
                assert x['dataset_index'] == g['index'] == i
                assert x['problem'] == row['problem'] and str(x['gold']) == str(row['answer'])
                ids = x['token_ids']
                assert len(ids) == x['tokens']
                think = ids.index(151649) if 151649 in ids else len(ids)
                assert think == x['thinking_tokens']
                compact.append(dict(dataset_index=i, problem_sha256=row['problem_sha256'],
                    correct=g['author_correct'], tokens=len(ids), thinking_tokens=think,
                    all_branch_output_tokens=len(ids), budget_tokens_including_probe_prompt=len(ids),
                    capped=len(ids)==16000))
            groups[arm] = dict(records=compact, summary=plan['frozen_benchmarks'][role]['groups'][key])
        history[role] = dict(groups=groups, evaluation_sha256=sha(ep), grading_sha256=sha(gp),
                             evaluation_path=str(ep), grading_path=str(gp))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpu-authorized', action='store_true')
    args = p.parse_args()
    if not args.gpu_authorized:
        raise ValueError('Explicit authorization for this full-test batch required')
    plan, receipt = read(args.plan), read(args.receipt)
    validate(plan, receipt, args.plan)
    out = args.output_root/plan['run_id']; out.mkdir(parents=True, exist_ok=False)
    save(out/'resolved_plan.json', plan); save(out/'launch_receipt.json', receipt)
    save(out/'historical_reference.json', historical(plan))
    started = time.perf_counter(); results = {}; monitor = None
    try:
        check = subprocess.run([sys.executable, str(HERE/'test_native.py')],
                               capture_output=True, text=True, timeout=60)
        save(out/'native_cpu_check.json', dict(returncode=check.returncode,
                                              stdout=check.stdout, stderr=check.stderr))
        assert check.returncode == 0
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        os.environ['PYTHONNOUSERSITE'] = '1'
        sys.path[:0] = [str(ROOT/'sources/EasySteer/vllm-steer'),
                       str(ROOT/'sources/EasySteer'),
                       str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        a = plan['assets']
        tok = AutoTokenizer.from_pretrained(a['model_path'], local_files_only=True)
        boundaries = sorted(i for piece, i in tok.get_vocab().items() if 'ĊĊ' in piece)
        steer = SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',
            data=from_pt_direction(a['vector']['path'], layers=[20]), algorithm='rebalance',
            scale=1., layers=[20], normalize=False,
            apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(a['fit']['path'])['parameters'], boundary_token_ids=boundaries,
                        think_start_token_id=151648, think_end_token_id=151649))])
        llm = LLM(model=a['model_path'], dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=32768, max_num_seqs=256, max_num_batched_tokens=32768,
            gpu_memory_utilization=.9, enable_steer_vector=True, steer_algorithms=['rebalance'],
            steer_graph_mode='in_graph', enforce_eager=False, enable_chunked_prefill=False,
            enable_prefix_caching=False, async_scheduling=True, seed=42)
        core = llm.llm_engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        startup = time.perf_counter()-started
        save(out/'runtime_identity.json', dict(python=sys.version, torch=torch.__version__, vllm_path=__import__('vllm').__file__, plan_sha256=sha(args.plan)))
        for role in ROLES:
            rows=plan['datasets'][role]['rows']
            prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in rows]
            assert max(map(len,prompts))+16000<32768
            arm='RCnegative'
            folder=out/role/arm; folder.mkdir(parents=True)
            setup_start = time.perf_counter()
            mode = 'negative'
            adapter = Adapter(llm, tok, mode=mode, gate_on=True)
            setup = time.perf_counter()-setup_start
            original_preempt = core.scheduler._preempt_request
            core.scheduler._preempt_request = Adapter.reject_preempt
            generated = {}; io_seconds = 0.
            gpu_stream = (folder/'gpu.csv').open('x', encoding='utf8')
            monitor = subprocess.Popen(['nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits', '--loop-ms=1000'],
                stdout=gpu_stream, stderr=subprocess.DEVNULL)
            torch.cuda.synchronize(); began = time.perf_counter()
            start_utc = time.time()
            save(folder/'start.json', dict(unix_seconds=start_utc, setup_seconds=setup))
            try:
                ids = llm.enqueue([dict(prompt_token_ids=x) for x in prompts],
                    sampling_params=SamplingParams(temperature=.7, top_p=.95, seed=42,
                        max_tokens=16000, skip_special_tokens=False),
                    steering=steer, use_tqdm=False)
                ops = llm.llm_engine.output_processor.request_states
                mapping = {ops[rid].external_req_id:rid for rid in ids}
                rowmap = dict(zip(ids, rows))
                save(folder/'request_mapping.json', {rid:dict(train_index=r['train_index'],
                    problem_sha256=r['problem_sha256']) for rid,r in rowmap.items()})
                with (folder/'partial.jsonl').open('x', encoding='utf8') as stream:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.perf_counter()-began > plan['arm_ceiling_seconds']:
                            raise TimeoutError('Fixed arm limit reached; partials retained')
                        for output in llm.llm_engine.step():
                            assert output.finished, 'Native FINAL_ONLY output required'
                            rid = mapping[output.request_id]; answer = output.outputs[0]
                            assert rid not in generated
                            ts = list(answer.token_ids)
                            rec = dict(rowmap[rid], request_id=rid, token_ids=ts,
                                tokens=len(ts), thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),
                                text=tok.decode(ts, skip_special_tokens=True),
                                finish_reason=answer.finish_reason)
                            assert len(ts) <= 16000
                            generated[rid] = rec
                            write_start = time.perf_counter()
                            stream.write(json.dumps(rec, ensure_ascii=False)+'\n'); stream.flush()
                            io_seconds += time.perf_counter()-write_start
                    llm.llm_engine.step()
                    for rid in ids:
                        if rid in runner.req_states.req_id_to_index:
                            runner._remove_request(rid)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-began
                assert len(generated) == len(rows)
                results[role] = dict(status='complete', records=[generated[rid] for rid in ids],
                    generation_seconds=seconds, setup_seconds=setup, checkpoint_io_seconds=io_seconds,
                    generation_start_unix=start_utc, generation_end_unix=time.time(),
                    events=adapter.completed if adapter.enabled else {},
                    probe_count=0, extra_model_forward_count=0,
                    control_gpu_seconds=None,
                    timing_note='End-to-end generation includes required bookkeeping and partial I/O; I/O is overlapping subset, not additive. Control kernels not isolated; no speedup claimed from small pilot.')
                save(folder/'result.json', results[role])
                print(json.dumps(dict(dataset=role, completed=len(rows), generation_seconds=seconds)), flush=True)
            finally:
                monitor.terminate(); monitor.wait(timeout=10); monitor=None; gpu_stream.close()
                core.scheduler._preempt_request = original_preempt
                adapter.close()
        save(out/'batch_status.json', dict(status='complete', datasets=ROLES,
            startup_seconds=startup, wall_seconds=time.perf_counter()-started,
            plan_sha256=sha(args.plan), receipt_sha256=sha(args.receipt)))
    except BaseException as exc:
        if monitor is not None:
            monitor.terminate(); monitor.wait(timeout=10)
        save(out/'failure.json', dict(error=repr(exc), completed_arms=list(results),
                                    wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()
