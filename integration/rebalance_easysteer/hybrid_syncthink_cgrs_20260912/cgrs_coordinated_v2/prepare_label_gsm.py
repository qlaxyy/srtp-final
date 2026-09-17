"""Freeze the user-authorized single L27/L27 GSM8K transfer, without tuning."""
import hashlib,json,tarfile,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE/'label_alignment_20260917'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def phash(s):return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()
def main():
    oldpath=next((ROOT/'.codex_work/cgrs_v2_full_run1_verified').rglob('resolved_plan.json'))
    old=read(oldpath);rows=old['datasets']['gsm8k_test']['rows'];assert len(rows)==1319
    source=ROOT/'sources/ReBalance/Data/Math_GSM8K/test.jsonl'
    original=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    hashes=[phash(r['problem']) for r in rows]
    assert hashes==[phash(r['question'] if 'question' in r else r['problem']) for r in original]
    assert len(set(hashes))==1319 and hashes==[r['problem_sha256'] for r in rows]
    frozen=ROOT.parent/'srtp-final/.codex_work/auto_code_v2_500_20260908'
    ep=frozen/'gsm8k_eval.json';gp=frozen/'gsm8k_author_grading.json';ev=read(ep);grading=read(gp)
    expected=old['frozen_benchmarks']['gsm8k_test']['artifacts']
    assert sha(ep)==expected['evaluation_sha256'] and sha(gp)==expected['grading_sha256']
    for k in ('seed','temperature','top_p','max_tokens','max_model_len'):assert ev['protocol'][k]==old['runtime'][k]
    groups={}
    for name,key in [('U','baseline'),('R','rebalance_dynamic')]:
        records=[]
        for i,(r,l) in enumerate(zip(ev[key]['records'],grading['groups'][key]['records'])):
            assert r['dataset_index']==l['index']==i and phash(r['problem'])==hashes[i]
            records.append(dict(dataset_index=i,problem_sha256=hashes[i],correct=l['author_correct'],tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
        assert len(records)==1319
        groups[name]=dict(records=records,generation_seconds=ev[key]['summary']['generation_seconds'])
    rcpath=oldpath.parent/'gsm8k_test/RCnegative/result.json';rc=read(rcpath)
    lp=rcpath.parent/'author_partial.jsonl';labels=[json.loads(s) for s in lp.read_text().splitlines()]
    assert rc['status']=='complete' and len(rc['records'])==len(labels)==1319
    labels={l['dataset_index']:l for l in labels};rcrows={r['dataset_index']:r for r in rc['records']};records=[]
    for i in range(1319):
        r=rcrows[i];l=labels[i]
        assert phash(r['problem'])==r['problem_sha256']==l['problem_sha256']==hashes[i]
        assert hashlib.sha256(r['text'].encode()).hexdigest()==l['text_sha256']
        records.append(dict(dataset_index=i,problem_sha256=hashes[i],correct=l['correct'],tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
    groups['RC14']=dict(records=records,generation_seconds=rc['generation_seconds'])
    refs=dict(groups=groups,grader_sha256={n:grading[n+'_sha256'] for n in ('grader.py','parser.py')},source_sha256={str(p):sha(p) for p in (oldpath,source,ep,gp,rcpath,lp)})
    save(OUT/'historical_gsm1319.json',refs)
    prior=read(OUT/'plan_v2_math500.json')
    plan=dict(run_id='label_alignment_gsm1319_20260917_run1',dataset_key='gsm8k',seed=42,max_new_tokens=16000,rows=rows,arms=[prior['arms'][0]],runtime_target=prior['runtime_target'],decision=prior['decision'],hard_stop_seconds_per_arm=900,process_hard_stop_seconds=1200,scope='Frozen MATH-selected L27/L27 transferred unchanged to all GSM8K; one seed, one arm, historical U/R/RC14 only; no tuning.',expected_generation_seconds=[180,360])
    plan['decision']['multiplicity']='One transferred candidate; prior selection and exposed benchmarks disclosed.'
    save(OUT/'plan_gsm1319.json',plan)
    release=read(OUT/'release_v2.json')
    release.update(status='Authorized GSM8K transfer; reuse passed native engineering and unchanged mechanism',plan_relative_path='label_alignment_20260917/plan_gsm1319.json',plan_sha256=sha(OUT/'plan_gsm1319.json'),references_artifact='historical_gsm1319.json',engineering_parent_release_sha256=sha(OUT/'release_v2.json'),engineering_complete_sha256=sha(ROOT/'.codex_work/label_alignment_20260917/engineering_run1/complete.json'))
    release['artifact_sha256'].update({'plan_gsm1319.json':sha(OUT/'plan_gsm1319.json'),'historical_gsm1319.json':sha(OUT/'historical_gsm1319.json')})
    release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in release['source_sha256']}
    save(OUT/'release_gsm1319.json',release)
    import run_label_alignment
    run_label_alignment.validate(plan,release,'full')
    oldrelease=read(OUT/'release_v2.json')
    assert all(release['source_sha256'][n]==h for n,h in oldrelease['source_sha256'].items() if n not in ('run_label_alignment.py','grade_label_alignment.py'))
    target=ROOT/'.codex_work/label_alignment_20260917/gsm1319_deploy_run1.tar'
    with tarfile.open(target,'x') as t:
        for n in release['source_sha256']:t.add(HERE/n,arcname=n)
        t.add(OUT,arcname=OUT.name)
    print(json.dumps(dict(tar_sha256=sha(target),references={k:dict(count=len(v['records']),correct=sum(r['correct'] for r in v['records']),generation_seconds=v['generation_seconds']) for k,v in groups.items()}),indent=2))
if __name__=='__main__':main()
