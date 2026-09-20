"""Package locked complete-response instrumentation; no generation or fitting."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',default='feedback_policy_replay_20260918_run1');ap.add_argument('--legacy-manifest',type=Path);args=ap.parse_args()
    home = Path(__file__).resolve().parent
    root = next(p for p in home.parents if (p/'.codex_work').is_dir())
    out = home/args.run_id
    out.mkdir(exist_ok=False)
    audit = home/'feedback_preference_audit_20260918_run1'
    proposal = json.loads((audit/'probe_proposal.json').read_text())
    release = json.loads((home/'repetition_shared_math500_20260918_run2/release.json').read_text())
    old = home/'self_feedback_train500_20260918_run1'
    original = json.loads((old/'plan.json').read_text())
    archive = root/'.codex_work/self_feedback_train500_20260918_evidence.tar.gz'
    assert sha(archive.read_bytes()) == 'cd38a0f003ed26d53037dbc9c74483e1943d8032bf150ae237e8cdb8c2b443ca'
    with tarfile.open(archive) as tf:
        l27 = json.load(tf.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    sources = dict(U=json.loads((old/'baseline.json').read_text()), L27=l27)
    rows = []
    for item in proposal['questions']:
        q = item['question']
        for source in ['U', 'L27']:
            r = sources[source][q]
            assert r['problem_sha256'] == item['problem_sha256']
            assert r['token_ids'][-1] == 151643 and 151649 in r['token_ids']
            rows.append(dict(key=str(q)+'_'+source, question=q, source=source,
                problem_sha256=r['problem_sha256'], chosen=source == item['chosen_source'],
                prompt_token_ids=original['rows'][q]['prompt_token_ids'], token_ids=r['token_ids']))
    assert sum(len(r['prompt_token_ids'])+len(r['token_ids']) for r in rows) == 43957
    files = {}
    names = ['run_feedback_replay.py','feedback_replay_capture.py','motivation_replay_sampler.py',
             'adapter.py','policy.py','label_alignment_adapter.py','test_feedback_capture_cpu.py']
    for name in names:
        raw = (home/name).read_bytes().replace(b'\r\n', b'\n')
        ast.parse(raw)
        if name in release['source_sha256']:
            assert sha(raw) == release['source_sha256'][name]
        files[name] = raw
    files['rows.json'] = (json.dumps(rows, separators=(',', ':'))+'\n').encode()
    files['opening.npz'] = (home/'repetition_shared_math500_20260918_run2/opening.npz').read_bytes()
    assert sha(files['opening.npz']) == release['artifact_sha256']['opening.npz']
    plan = dict(status='Locked engineering-only batch before GPU; no preference training',
        assets=release['assets'], runtime_root=release['runtime_root'], runtime_source_sha256=release['runtime_source_sha256'],
        input_sha256={n:sha(b) for n,b in files.items()},
        purpose='Compare legacy saved-token forcer with scoring instrumentation through identical frozen native L27. Capture full response logp, raw confidence, lexical state/gate and per-input native coefficient history.',
        groups=['legacy_forcer','scoring_capture'], questions=8, saved_responses=16, full_forward_token_inputs_per_group=43957,
        new_answers=0, seed=42, temperature=0, force_saved_tokens=True,
        scoring='Temperature1 full vocabulary after L27 penalty, no top-p; sampled argmax is overwritten by saved target before lexical/native controller acceptance',
        runtime=dict(max_num_seqs=16,max_model_len=16384,max_num_batched_tokens=32768,gpu_memory_utilization=.9,async_scheduling=True),
        expected_minutes=[3,10], process_ceiling_seconds=900,
        stop=['Any forced token mismatch','Any raw confidence or complete native history mismatch','KV preemption','Missing/nonfinite trace','Source/model/asset hash mismatch','900s ceiling'],
        scope='No residual hook, backward, optimizer, benchmark, new answer or efficacy claim. Native trace is a prerequisite to subsequent fixed-history HF gradient engineering, not proof of HF/vLLM equivalence.',
        data_registry=[dict(question=r['question'],source=r['source'],problem_sha256=r['problem_sha256'],purpose='Existing development training answer replay') for r in rows])
    if args.legacy_manifest:
        plan['legacy_reference']=dict(directory='/root/autodl-tmp/results/feedback_policy_replay_20260918_run1',
            manifest_sha256=sha(args.legacy_manifest.read_bytes()),seconds=81.53113169968128,
            original_plan_sha256='9811b44952a356c04368cab4b83a47dd0b69f13abcac3deacf1e4f07632d9648')
        plan['groups']=['scoring_capture'];plan['purpose']+=' Reuse completed original legacy pass; preserve failed wrapper run.'
    files['plan.json'] = (json.dumps(plan, ensure_ascii=False, indent=2)+'\n').encode()
    for name, raw in files.items():
        (out/name).write_bytes(raw)
    (out/'.gitignore').write_text('rows.json\n*.py\npackage.tar.gz\n', encoding='ascii')
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n', encoding='ascii')
    with tarfile.open(out/'package.tar.gz','w:gz') as tf:
        for name, raw in files.items():
            item=tarfile.TarInfo(name); item.size=len(raw); tf.addfile(item, io.BytesIO(raw))
    print(json.dumps(dict(directory=str(out), package_sha256=sha((out/'package.tar.gz').read_bytes()),plan_sha256=sha(files['plan.json']), saved_tokens=sum(len(r['token_ids']) for r in rows))))


if __name__ == '__main__':
    main()
