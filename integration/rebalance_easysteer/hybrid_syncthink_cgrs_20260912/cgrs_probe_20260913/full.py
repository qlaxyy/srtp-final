"""Fixed RC-only full evaluation. Reads frozen baselines; never regenerates them."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from run import HERE, ROOT, build_engine, deployment_record, read, run_arm, save, sha, validate


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    assert list(plan['datasets']) == ['math_test', 'gsm8k_test']
    assert plan['arms'] == ['RC']
    for sub in plan['datasets'].values():
        assert sub['phase'] == 'full_test' and sub['fast_io'] and sub['main_window'] == 256
        validate(sub)
    output = args.output_root/plan['run_id']
    output.mkdir(parents=True, exist_ok=False)
    save(output/'resolved_plan.json', plan)
    save(output/'historical_reference.json', historical(plan))
    started = time.monotonic()
    monitor = None
    with (output/'gpu_utilization.csv').open('x') as gpu:
        try:
            monitor = subprocess.Popen(['nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw',
                '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu, stderr=subprocess.DEVNULL)
            check = subprocess.run([sys.executable, str(HERE/'test_native.py')],
                                    capture_output=True, text=True, timeout=60)
            save(output/'native_cpu_check.json', dict(returncode=check.returncode,
                stdout=check.stdout, stderr=check.stderr))
            if check.returncode:
                raise RuntimeError('Native CPU gate failed')
            first = plan['datasets']['math_test']
            llm, tok, steering, boundaries = build_engine(first)
            startup = time.monotonic()-started
            import torch, vllm
            save(output/'runtime_identity.json', dict(python=sys.version, torch=torch.__version__,
                vllm=vllm.__version__, vllm_path=vllm.__file__, deployment=deployment_record(),
                plan_sha256=sha(args.plan), assets=first['assets'], source_sha256=first['source_sha256'],
                timezone_offset_seconds=-time.altzone if time.daylight and time.localtime().tm_isdst else -time.timezone))
            for role, sub in plan['datasets'].items():
                folder = output/role
                folder.mkdir()
                run_arm(llm, tok, steering, boundaries, sub['rows'], 'RC', sub, folder)
                print('COMPLETED', role, flush=True)
            save(output/'batch_status.json', dict(status='complete', startup_seconds=startup,
                                                  wall_seconds=time.monotonic()-started))
        except BaseException as exc:
            save(output/'failure.json', dict(status='failed', error=repr(exc),
                                            wall_seconds=time.monotonic()-started))
            raise
        finally:
            if monitor:
                monitor.terminate()
                monitor.wait(timeout=10)


if __name__ == '__main__':
    main()
