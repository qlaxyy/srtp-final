"""Freeze one outcome-conditioned vector iteration on existing calibration parents."""
import ast,hashlib,io,json,tarfile,unicodedata
from pathlib import Path
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    run='self_feedback_train500_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n*.pt binary -text\n')
    (out/'.gitignore').write_text('baseline.json\npackage.tar.gz\ndownloaded/\n')
    archive=ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    train=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    with tarfile.open(archive) as t:
        manifest=json.load(t.extractfile('manifest.json'));raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    assert sha(train)==manifest['train_sha256'] and manifest['temperature']==0
    generations=[json.loads(s) for s in raw.splitlines()];gold=[json.loads(s) for s in train.read_text(encoding='utf8').splitlines()]
    rows=[];baseline=[]
    norm=lambda s:''.join(unicodedata.normalize('NFKC',s).split())
    forbidden=set()
    for name in ['Math_Math500','Math_GSM8K','Math_AIME2024','Math_AIME2025','Math_AMC23','Math_Olympiad']:
        for s in (ROOT/'sources/ReBalance/Data'/name/'test.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(s);forbidden.add(norm(r.get('problem',r.get('question',''))))
    overlaps=[]
    for i,r in enumerate(generations):
        g=gold[r['train_index']];assert r['problem']==g['problem']
        identity=dict(dataset_index=i,train_index=r['train_index'],problem=r['problem'],
            problem_sha256=hashlib.sha256(norm(r['problem']).encode()).hexdigest(),split='train')
        if norm(r['problem']) in forbidden:overlaps.append(i)
        rows.append(dict(g,**identity,prompt_token_ids=r['prompt_token_ids']))
        baseline.append(dict(identity,token_ids=r['token_ids'],text=r['text'],finish_reason=r['finish_reason']))
    save(out/'baseline.json',baseline)
    previous=HERE/'question_centered_math_20260918_run1';rel=read(previous/'release.json');plan=read(previous/'plan.json')
    protocol=dict(method='Outcome-conditioned original class means, one iteration only',
        input_generation_member_sha256=hashlib.sha256(raw).hexdigest(),
        label='Eligible parent iff U and L27 both author-correct, both naturally closed, L27 thinking AND total tokens strictly smaller, and no benchmark overlap',
        benchmark_overlap_excluded_indices=overlaps,
        extraction='Reuse ONLY original unsteered saved step states, original OR over/AND under labels and old quantiles; restrict both classes to eligible parents; preserve original step weighting; raw mean_over-minus-mean_under then match original vector norm',
        intent='First minimal outcome-feedback candidate; not direct long-short hidden-state contrast, not gold per-step overthinking labels',
        incorrect_short='Record all correctness flips and harmful shortening; never treat wrong-short states as desired target',
        confidence_variance='No additional confidence/variance hard gate in first candidate; no thresholds tuned on outcome',
        gate='At least30 eligible parents contributing to each class; finite nonzero vector; exact old vector reproduction; keep failures and caps in collection report',
        unchanged=['decoder output layer20','old fit and dynamic curve','L27 suppression','old vector norm'],
        stop='No repeated iterations or alternative masks if support fails. New candidate evaluated later against frozen L27 on separate questions; calibration results are not efficacy evidence.',
        historical_limit='Greedy settings and exact original prompt IDs matched, but historical batch/runtime numerics can differ; pairs are observed preferences, not causal labels',
        coordination='Original500 calibration parents already shared training assets; no new holdout claimed. Existing test275+11 pairs are excluded from all fitting.')
    save(out/'protocol.json',protocol)
    plan.update(run_id=run,experiment_kind='self_feedback_collection_v1',dataset='Math_Train calibration500',rows=rows,
        single_transfer=True,temperature=0,top_p=1,arms=[dict(name='L27_L27',suppression_table=run+'/opening.npz')],
        change='Collect L27 greedy outputs on original500 calibration prompts; reuse all original U answers; no evaluation or vector change yet',
        decision=protocol,advancement=protocol['gate'],reference_files={},new_answers=500,
        hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=2100,
        expected_cost='One greedy500 training collection,10-20min estimate; stop at1800s; no probes or hidden-state forward required for first candidate')
    save(out/'plan.json',plan);(out/'opening.npz').write_bytes((previous/'opening.npz').read_bytes())
    sources=list(rel['source_sha256'])+['prepare_self_feedback.py','fit_self_feedback.py']
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in sources}
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    rel.update(status='Training collection, no efficacy claim',artifact_root=run,engineering_rows=rows[:8],
        plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ['opening.npz','baseline.json','protocol.json','plan.json']},
        source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    save(out/'release.json',rel)
    for n in ['release.json','plan.json','protocol.json','baseline.json','opening.npz']:files[run+'/'+n]=(out/n).read_bytes()
    with tarfile.open(out/'package.tar.gz','w:gz') as t:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(rows=len(rows),benchmark_overlap_excluded=overlaps,package_sha256=sha(out/'package.tar.gz'))))

if __name__=='__main__':main()
