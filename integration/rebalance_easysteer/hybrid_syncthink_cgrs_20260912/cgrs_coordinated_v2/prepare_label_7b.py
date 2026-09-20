"""Fixed 7B transfer, independent frozen calibration and historical controls."""
from prepare_label_gsm import read,sha,save,phash,HERE,ROOT,OUT
import json,hashlib,tarfile

def main():
    op=next(p for p in (ROOT/'.codex_work/cgrs_v2_7b_run1_verified').rglob('resolved_plan.json') if 'full_7b' in str(p))
    old=read(op);history=read(op.parent/'historical_reference.json')
    base=read(OUT/'release_v2.json');mainwork=ROOT.parent/'srtp-final/.codex_work'
    assert old['assets']['model_files']['tokenizer.json']==base['assets']['model_files']['tokenizer.json']
    new_releases=[]
    for role,dataset,count,seqs in [('math_test','math',500,32),('gsm8k_test','gsm8k',1319,64)]:
        rows=old['datasets'][role]['rows'];assert len(rows)==count
        src=ROOT/('sources/ReBalance/Data/Math_Math500/test.jsonl' if dataset=='math' else 'sources/ReBalance/Data/Math_GSM8K/test.jsonl')
        originals=[json.loads(s) for s in src.read_text(encoding='utf8').splitlines()]
        hashes=[phash(r['problem']) for r in rows]
        assert hashes==[phash(r.get('problem',r.get('question'))) for r in originals] and len(set(hashes))==count
        meta=old['frozen_benchmarks'][role]['artifacts']
        folder=mainwork/('qwen7b_final_20260909/final_math500_completed_20260909' if dataset=='math' else 'qwen7b_validation/auto_code_v2_qwen7b_20260908/formal_kv_replay')
        ep=folder/meta['evaluation'];gp=folder/meta['grading'];assert sha(ep)==meta['evaluation_sha256'] and sha(gp)==meta['grading_sha256']
        ev=read(ep);grades=read(gp);groups={}
        for name,key in [('U','baseline'),('R','rebalance_dynamic')]:
            records=[]
            for i,(r,l) in enumerate(zip(ev[key]['records'],grades['groups'][key]['records'])):
                assert r['dataset_index']==l['index']==i and phash(r['problem'])==hashes[i]
                records.append(dict(dataset_index=i,problem_sha256=hashes[i],correct=l['author_correct'],tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
            assert len(records)==count
            groups[name]=dict(records=records,generation_seconds=old['frozen_benchmarks'][role]['groups'][key]['generation_seconds'])
        rp=op.parent/role/'RCnegative/result.json';lp=rp.parent/'author_partial.jsonl';rc=read(rp)
        labels={r['dataset_index']:r for r in map(json.loads,lp.read_text().splitlines())};rs={r['dataset_index']:r for r in rc['records']}
        assert rc['status']=='complete' and len(labels)==len(rs)==count
        records=[]
        for i in range(count):
            r=rs[i];l=labels[i]
            assert phash(r['problem'])==r['problem_sha256']==l['problem_sha256']==hashes[i]
            assert hashlib.sha256(r['text'].encode()).hexdigest()==l['text_sha256']
            records.append(dict(dataset_index=i,problem_sha256=hashes[i],correct=l['correct'],tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
        groups['RC14']=dict(records=records,generation_seconds=rc['generation_seconds'])
        refs=dict(groups=groups,grader_sha256={n:grades[n+'_sha256'] for n in ('grader.py','parser.py')},source_sha256={str(p):sha(p) for p in (op,src,ep,gp,rp,lp)})
        refname=f'historical_7b_{dataset}.json';save(OUT/refname,refs)
        prior=read(OUT/'plan_gsm1319.json')
        plan=dict(prior,run_id=f'label_alignment_7b_{dataset}_20260917_run1',dataset_key=dataset,single_transfer=True,rows=rows,
            hard_stop_seconds_per_arm=4200,process_hard_stop_seconds=4800,
            execution=dict(sync_replay=True,max_model_len=17920,max_num_seqs=seqs,max_num_batched_tokens=4096,gpu_memory_utilization=.95,chunked_prefill=True),
            scope='Unchanged L27 opening suppression transferred to independently calibrated 7B; one arm and seed; reuse U/R/RC14.',
            expected_generation_seconds=[1500,2400] if dataset=='math' else [600,1200])
        plan['runtime_target']=dict(plan['execution'],dtype='bfloat16',seed=42,temperature=.7,top_p=.95,max_tokens=16000,async_scheduling=False)
        pn=f'plan_7b_{dataset}.json';save(OUT/pn,plan)
        release=read(OUT/'release_v2.json')
        release.update(status='User authorized 7B transfer; new synchronous lexical replay engineering required',assets=old['assets'],plan_relative_path='label_alignment_20260917/'+pn,plan_sha256=sha(OUT/pn),references_artifact=refname)
        release['artifact_sha256'].update({pn:sha(OUT/pn),refname:sha(OUT/refname)})
        release['source_sha256'].update({n:'' for n in ('replay_adapter.py','replay_label_alignment.py')})
        release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in release['source_sha256']}
        if dataset=='gsm8k':
            release['engineering_parent_relative_path']='label_alignment_20260917/release_7b_math.json'
            release['engineering_parent_release_sha256']=sha(OUT/'release_7b_math.json')
        rn=f'release_7b_{dataset}.json';save(OUT/rn,release);new_releases.append(release)
        import run_label_alignment
        run_label_alignment.validate(plan,release,'full')
        print(dataset,{k:(len(v['records']),sum(r['correct'] for r in v['records'])) for k,v in groups.items()})
    target=ROOT/'.codex_work/label_alignment_20260917/7b_deploy_run1.tar'
    with tarfile.open(target,'x') as t:
        for n in new_releases[-1]['source_sha256']:t.add(HERE/n,arcname=n)
        t.add(OUT,arcname=OUT.name)
    print('package_sha256',sha(target))

if __name__=='__main__':main()
