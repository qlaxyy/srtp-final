"""CPU preparation on an isolated, patched checkout; requires executor receipt.

Does not execute a model forward. Never creates a reconciliation receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import run_batch as rb
from prepare_execution import phash


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coordination-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    plan = rb.read(HERE/'plan.json')
    proposal = rb.read(HERE/'data_usage.json')
    receipt = rb.read(a.coordination_receipt)
    assert receipt['data_usage_sha256'] == rb.sha(HERE/'data_usage.json')
    assert receipt['plan_sha256'] == rb.sha(HERE/'plan.json')
    assert receipt['cross_line_data_clear'] is True
    assert receipt['shared_code_integrated_in_isolated_checkout'] is True
    assert receipt['gpu_batch_authorized'] is True
    assert receipt['run_id'] == plan['run_id']
    assert receipt['authority'] and receipt['checked_registries']
    manifest = rb.read(HERE/'inference_manifest.json')
    for name, hashes in manifest['files'].items():
        assert rb.sha(ROOT/name) == hashes, name
    original = rb.read(HERE.parent/'experiment_plan.json')
    for name, meta in original['assets']['model_files'].items():
        assert rb.sha(Path(original['assets']['model_path'])/name) == meta['sha256']
    for key in ('vector','fit'):
        assert rb.sha(original['assets'][key]['path']) == original['assets'][key]['sha256']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader'], text=True)
    assert not apps.strip(), 'Shared GPU busy'
    reuse = rb.read(HERE/'engineering_reuse.json')
    source = Path(reuse['source'])
    for name, expected in reuse['files'].items():
        assert rb.sha(source/name) == expected, 'Reused output mismatch: '+name
    assert rb.read(source/'batch_status.json')['status'] == 'complete'
    assert rb.read(source/'engineering_gate.json')['status'] == 'pass'
    assert reuse['assets'] == original['assets']
    for name, expected in reuse['inference_sha256'].items():
        assert rb.sha(ROOT/name) == expected, 'Inference dependency changed: '+name
    # Engineering outputs are copied and hash-checked, with no new generation.
    rows = {'engineering': rb.dataset('engineering')}
    for role in plan['roles']:
        path = HERE.parent/'full_tests_20260913'/(role+'.jsonl')
        assert rb.sha(path) == proposal['question_files'][role]
        rows[role] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows[role]) == plan['counts'][role]
    history={};folder=Path('/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908')
    for role,b in zip(plan['roles'],plan['frozen_benchmarks']):
        assert rb.sha(HERE.parent/'full_tests_20260913'/(role+'.jsonl'))==plan['files'][role]
        meta=b['artifacts'];ep=folder/meta['evaluation'];gp=folder/meta['grading']
        assert rb.sha(ep)==meta['evaluation_sha256'];assert rb.sha(gp)==meta['grading_sha256']
        evaluation=rb.read(ep);grades=rb.read(gp)
        assert grades['input_sha256']==rb.sha(ep)
        for key in ('parser.py','grader.py'):assert rb.sha(ROOT/'sources/ReBalance/utils'/key)==grades[key+'_sha256']
        protocol=evaluation['protocol']
        for key,value in dict(max_tokens=16000,max_model_len=32768,temperature=.7,top_p=.95,seed=42,easysteer_output_layer=20).items():assert protocol[key]==value,(key,protocol[key])
        assert evaluation['provenance']['vector_sha256']==original['assets']['vector']['sha256']
        assert evaluation['provenance']['calibration_fit_sha256']==original['assets']['fit']['sha256']
        groups={}
        for arm,key in [('U','baseline'),('R','rebalance_dynamic')]:
            records=evaluation[key]['records'];labels=grades['groups'][key]['records'];assert len(records)==len(labels)==plan['counts'][role]
            compact=[]
            for i,(x,g,row) in enumerate(zip(records,labels,rows[role])):
                assert x['dataset_index']==g['index']==i and x['problem']==row['problem'] and str(x['gold'])==str(row['answer'])
                ids=x['token_ids'];assert len(ids)==x['tokens']
                think=ids.index(151649) if 151649 in ids else len(ids);assert think==x['thinking_tokens']
                compact.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=g['author_correct'],tokens=len(ids),thinking_tokens=think,capped=len(ids)==16000))
            assert sum(x['correct'] for x in compact)==b['groups'][key]['correct']
            groups[arm]=dict(records=compact,summary=b['groups'][key])
        history[role]=dict(groups=groups,evaluation_path=str(ep),evaluation_sha256=rb.sha(ep),grading_path=str(gp),grading_sha256=rb.sha(gp),protocol=protocol)
    rb.save(ROOT/'historical_mix05_full_reference.json',history)

    import torch
    import vllm
    from transformers import AutoTokenizer
    from rebalance_static_eval import build_prompt
    tok = AutoTokenizer.from_pretrained(original['assets']['model_path'], local_files_only=True)
    prompts = {role: [tok.encode(build_prompt(tok, r['problem'])) for r in group]
               for role, group in rows.items()}
    assert max(len(x) for group in prompts.values() for x in group) + 16000 < 32768
    sources = {}
    for folder in ('sources/EasySteer/vllm-steer/vllm', 'sources/EasySteer/easysteer',
                   'sources/ReBalance/utils', 'integration/rebalance_easysteer/eval',
                   'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.jsonl'):
                sources[path.relative_to(ROOT).as_posix()] = rb.sha(path)
    assert rb.sha(ROOT/'baseline_model_runner.py') == '88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5'
    sources['baseline_model_runner.py'] = rb.sha(ROOT/'baseline_model_runner.py')
    rb.save(a.output, dict(run_id=plan['run_id'], data_frozen=True,
        expansion=plan, soft2_rows=rows, mix05=True, source_sha256=sources,
        completed_engineering_reuse=reuse,
        historical_reference_sha256=rb.sha(ROOT/'historical_mix05_full_reference.json'),
        plan_sha256=rb.sha(HERE.parent/'experiment_plan.json'),
        mix05_plan_sha256=rb.sha(HERE/'plan.json'),
        coordination_receipt=receipt, coordination_sha256=rb.sha(a.coordination_receipt),
        assets=original['assets'], prompts=prompts,
        deployment=rb.read(ROOT/'deployment_identity.json'),
        environment=dict(build_tools=rb.check_build_tools(), torch=torch.__version__, vllm=vllm.__version__),
        output='/root/autodl-tmp/results/easysteer/'+plan['namespace']+'/'+plan['run_id']))
    print('PREPARED', rb.sha(a.output))


if __name__ == '__main__':
    main()
