"""Lock development-only source-stratified split and full saved-answer replay."""
import ast, hashlib, io, json, tarfile
from collections import Counter
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    home=Path(__file__).resolve().parent
    root=next(p for p in home.parents if (p/'.codex_work').is_dir())
    out=home/'feedback_expanded_20260919_run1';out.mkdir(exist_ok=False)
    audit=home/'feedback_preference_audit_20260918_run1'
    parents=json.loads((audit/'parents.json').read_text())
    engineer={185,215,105,481,349,55,127,301}
    selected=[p for p in parents if p['question'] not in engineer|{48,390}]
    comp=[p for p in selected if p['kind']=='correct_short_over_correct_long']
    guard=[p for p in selected if p not in comp]
    assert len(comp)==221 and len(guard)==5
    for source in ['U','L27']:
        group=sorted([p for p in comp if p['chosen_source']==source],key=lambda p:sha(('feedback-dev-v1:'+p['problem_sha256']).encode()))
        cut=int(.8*len(group))
        for i,p in enumerate(group): p['split']='fit' if i<cut else 'development_check'
    for p in guard:p['split']='correctness_guard_only'
    assert len({p['problem_sha256'] for p in selected})==226
    selected.sort(key=lambda p:p['question'])
    archive=root/'.codex_work/self_feedback_train500_20260918_evidence.tar.gz'
    assert sha(archive.read_bytes())=='cd38a0f003ed26d53037dbc9c74483e1943d8032bf150ae237e8cdb8c2b443ca'
    with tarfile.open(archive) as tf: l27=json.load(tf.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    old=home/'self_feedback_train500_20260918_run1'
    original=json.loads((old/'plan.json').read_text())
    sources={'U':json.loads((old/'baseline.json').read_text()),'L27':l27}
    rows=[]
    for p in selected:
        for source in ['U','L27']:
            r=sources[source][p['question']]
            assert r['problem_sha256']==p['problem_sha256']
            assert r['token_ids'][-1]==151643
            side='chosen' if source==p['chosen_source'] else 'rejected'
            assert len(r['token_ids'])==p['endpoints'][side]['total_tokens']
            rows.append(dict(key=str(p['question'])+'_'+source,question=p['question'],source=source,chosen=side=='chosen',split=p['split'],problem_sha256=p['problem_sha256'],prompt_token_ids=original['rows'][p['question']]['prompt_token_ids'],token_ids=r['token_ids']))
    split=dict(parents=selected,engineering_excluded=sorted(engineer),total_not_improved_excluded=[48,390],counts=dict(Counter(p['split'] for p in selected)),source_counts=dict(Counter(p['split']+':'+p['chosen_source'] for p in selected)),algorithm='sha256(feedback-dev-v1:+problem_sha256), within chosen-source, floor80pct fit',scope='All previously exposed original training parents; development only, never independent confirmation',analysis_precommit=['Compute pair gradient chosen-minus-rejected, full summed logp primary','Fit mean weighted equally per parent; also equal-source means as diagnostic only','Report cosine and sign of fit-direction projection separately for both development source groups and correctness guards','Do not optimize delta, select strengths, or run benchmarks in this batch','If either development source group has nonpositive median projection or any guard has negative projection, do not advance pooled direction; retain L27','Positive linear projections are not nonlinear likelihood or generation improvement'])
    release=json.loads((home/'feedback_policy_replay_20260918_run2/plan.json').read_text())
    names=['run_feedback_replay.py','feedback_replay_capture.py','motivation_replay_sampler.py','adapter.py','policy.py','label_alignment_adapter.py','run_feedback_gradient_probe.py','feedback_fixed_history.py']
    files={n:(home/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    for n,b in files.items():ast.parse(b)
    files['rows.json']=(json.dumps(rows,separators=(',',':'))+'\n').encode()
    files['split.json']=(json.dumps(split,indent=2)+'\n').encode()
    files['opening.npz']=(home/'feedback_policy_replay_20260918_run2/opening.npz').read_bytes()
    plan={k:release[k] for k in ['assets','runtime_root','runtime_source_sha256']}
    plan.update(input_sha256={n:sha(b) for n,b in files.items()},capture_only=True,runtime=dict(max_num_seqs=32,max_model_len=16384,max_num_batched_tokens=32768,gpu_memory_utilization=.9),questions=226,responses=len(rows),seed=42,new_answers=0,optimizer_steps=0,process_ceiling_seconds=1800,expected_minutes=[5,20],full_output_tokens=sum(len(r['token_ids']) for r in rows),full_context_tokens=sum(len(r['token_ids'])+len(r['prompt_token_ids']) for r in rows),max_context=max(len(r['token_ids'])+len(r['prompt_token_ids']) for r in rows),scope='Single capture through previously validated instrumentation; no second legacy replay, history_exact must be null. All complete answers, no truncation.',stop=['hash mismatch','forced token mismatch','raw confidence mismatch','KV preemption','nonfinite trace','OOM','1800 seconds'],split_sha256=sha(files['split.json']))
    files['plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    for n,b in files.items():(out/n).write_bytes(b)
    (out/'.gitignore').write_text('*.py\nrows.json\npackage.tar.gz\n')
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n')
    with tarfile.open(out/'package.tar.gz','w:gz') as tf:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);tf.addfile(item,io.BytesIO(b))
    print(json.dumps(dict(counts=split['counts'],sources=split['source_counts'],tokens=plan['full_output_tokens'],context=plan['full_context_tokens'],max_context=plan['max_context'],package_sha256=sha((out/'package.tar.gz').read_bytes()))))
if __name__=='__main__':main()
