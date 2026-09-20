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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coordination-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    plan = rb.read(HERE/'plan.json')
    proposal = rb.read(HERE/'data_proposal.json')
    receipt = rb.read(a.coordination_receipt)
    assert receipt['data_proposal_sha256'] == rb.sha(HERE/'data_proposal.json')
    assert receipt['plan_sha256'] == rb.sha(HERE/'plan.json')
    assert receipt['cross_line_data_clear'] is True
    assert receipt['shared_code_integrated_in_isolated_checkout'] is True
    assert receipt['gpu_batch_authorized'] is True
    assert receipt['run_id'] == plan['run_id']
    assert receipt['authority'] and receipt['checked_registries']
    manifest = rb.read(HERE/'patch_manifest.json')
    for name, hashes in manifest['files'].items():
        assert rb.sha(ROOT/name) == hashes['patched_lf_sha256'], name
    original = rb.read(HERE.parent/'experiment_plan.json')
    for name, meta in original['assets']['model_files'].items():
        assert rb.sha(Path(original['assets']['model_path'])/name) == meta['sha256']
    for key in ('vector','fit'):
        assert rb.sha(original['assets'][key]['path']) == original['assets'][key]['sha256']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader'], text=True)
    assert not apps.strip(), 'Shared GPU busy'
    # Reuse known eight engineering question inputs, not their old model outputs.
    rows = {'engineering': rb.dataset('engineering')}
    for role in plan['roles']:
        assert rb.sha(HERE/(role+'.json')) == proposal['question_files'][role]
        rows[role] = rb.read(HERE/(role+'.json'))
        assert len(rows[role]) == plan['counts'][role]
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
