"""CPU-only materialization of existing training data and source identities."""
import ast
import hashlib
import shutil
import subprocess
import tarfile
import unicodedata
from common import HOME, HERE, ROOT, read, save, sha, spans

def main():
    old = HERE / 'self_feedback_train500_20260918_run1'
    plan = read(old / 'plan.json')
    rows = plan['rows']
    assert len(rows) == 500 and len({r['train_index'] for r in rows}) == 500
    for r in rows:
        norm = ''.join(unicodedata.normalize('NFKC', r['problem']).split())
        assert hashlib.sha256(norm.encode()).hexdigest() == r['problem_sha256']
    archive = ROOT / '.codex_work/self_feedback_train500_20260918_evidence.tar.gz'
    assert sha(archive) == 'cd38a0f003ed26d53037dbc9c74483e1943d8032bf150ae237e8cdb8c2b443ca'
    with tarfile.open(archive) as t:
        import json
        records = json.load(t.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    assert len(records) == 500
    for r, source in zip(records, rows):
        assert r['problem_sha256'] == source['problem_sha256']
        r['prompt_token_ids'] = source['prompt_token_ids']
        assert len(r['token_ids']) <= 16000
    # Do not silently pool the old 404+96 collection with a newly generated arm.
    save(HOME / 'saved_l27.json', dict(records=records, source_sha256=sha(archive),
        provenance='Historical greedy collection: 404 seq256 + 96 recovery seq32; calibration reuse only'))
    save(HOME / 'rows.json', rows)
    save(HOME / 'registry.json', [dict(question=i, train_index=r['train_index'],
        problem_sha256=r['problem_sha256'], use='shared original calibration; never holdout') for i,r in enumerate(rows)])
    old_release = read(old / 'release.json')
    save(HOME / 'assets.json', old_release['assets'])
    save(HOME / 'runtime_hashes.json', old_release['runtime_source_sha256'])
    shutil.copyfile(old / 'opening.npz', HOME / 'opening.npz')
    config = dict(model='DeepSeek-R1-Distill-Qwen-1.5B', seed=42, max_new_tokens=16000,
        calibration_temperature=0, calibration_top_p=1, evaluation_temperature=.7, evaluation_top_p=.95,
        max_num_seqs=256, max_num_batched_tokens=32768, gpu_memory_utilization=.90,
        replay='unsteered BF16 SDPA; full-vocabulary raw pmax and raw block outputs on identical saved prefixes',
        labels='over = lexical_hit OR c<q25; under = NOT lexical_hit AND c>q75',
        variance='adjacent step confidence difference squared / 4; first step zero',
        vector='raw mean(over)-mean(under); original auto layer selection; no norm matching',
        function='reuse original calibrate_auto.select/fit unchanged; recompute all fitted parameters',
        iterations=1, new_training_answers=500, reused_training_answers=500,
        gpu_authorized=False, generation_ceiling_seconds=2100, replay_ceiling_seconds_per_source=2400,
        evaluation='Two full MATH500 candidates: R-derived calibration + R; L27-derived calibration + L27. No cross grid.',
        gate='Both token means decrease vs own frozen parent, accuracy loss <=2pp vs own parent, caps do not increase. Report all results vs L27 too. No threshold tuning.',
        exposed_benchmark=True, followup='No second iteration, GSM8K or7B until this round is reported',
        warning='This is distribution-refresh calibration of the base model, not on-policy state extraction or a convergence theorem.')
    save(HOME / 'plan.json', config)
    # Deterministic boundary checks including unfinished cap and consecutive delimiters.
    assert spans([1,9,2,151649], {9}) == ([(0,1),(2,3)], 3)
    assert spans([1,9,2], {9}) == ([(0,1)], 3)
    assert spans([9,9,2,151649], {9}) == ([(2,3)], 3)
    assert spans([151649], {9}) == ([],0)
    deps = ['integration/rebalance_easysteer/scripts/calibrate_auto.py',
            'integration/rebalance_easysteer/scripts/calibrate_own_vector.py',
            'sources/ReBalance/hidden_analysis_auto.py',
            'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py']
    deps += [str((HERE / n).relative_to(ROOT)).replace('\\','/') for n in
             ['adapter.py','label_alignment_adapter.py','policy.py','lexicon_automaton.py']]
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
    files = {p.name:sha(p) for p in HOME.iterdir() if p.is_file() and p.name not in ('manifest.json','cpu_checks.json')}
    save(HOME / 'manifest.json', dict(files=files, repo_files={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in deps},
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    from common import verify
    verify()
    save(HOME / 'cpu_checks.json', dict(passed=True, rows=500, unique_hashes=len({r['problem_sha256'] for r in rows}),
        reused_l27_records=500, boundary_cases=4, syntax=True, input_hashes=True,
        gpu_tests=False, limitations='Local syntax/data/contract checks only; vLLM/transformers replay requires GPU engineering gate.'))
    print('CPU preparation passed; GPU not contacted.')

if __name__ == '__main__':
    main()
