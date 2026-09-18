"""Fail-closed transfer preparation: only a predeclared MATH-pass may transfer."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
from prepare_length_vector import HERE,read,sha,save

def main():
    parent=HERE/'trajectory_length_vector_20260918_run2'
    decisions=[read(parent/(name+'_decision.json')) for name in ('LENGTH_L27','LENGTH_NORM_L27')]
    candidates=[d for d in decisions if d['eligible_without_uniform_schedule_recheck']]
    if not candidates:raise SystemExit('No eligible candidate; GSM transfer forbidden by fixed rule')
    chosen=min(candidates,key=lambda d:(d['summary']['mean_tokens'],-d['summary']['accuracy_percent']))
    name=chosen['arm'];oldrelease=read(parent/'release.json');oldplan=read(parent/'plan.json')
    root=HERE.parents[3];prior=root/'.codex_work/label_alignment_20260917/label_alignment_gsm1319_20260917_run1'
    complete=read(prior/'complete.json');analysis=read(prior/'analysis.json');priorplan=read(prior/'plan.json')
    assert complete['passed'] and complete['phase']=='full'
    data=read(prior/'L27_L27/result.json');raw=(prior/'L27_L27/author_partial.jsonl').read_bytes()
    labels=[json.loads(s) for s in raw.splitlines()]
    saved=analysis['candidates']['L27_L27'];assert sha(prior/'L27_L27/result.json')==saved['source_result_sha256']
    assert hashlib.sha256(raw).hexdigest()==saved['labels_sha256']
    refs=read(HERE/'label_alignment_20260917/historical_gsm1319.json');records=[]
    for i,(q,r,l) in enumerate(zip(priorplan['rows'],data['records'],labels)):
        assert q['dataset_index']==r['dataset_index']==l['dataset_index']==i
        assert q['problem_sha256']==r['problem_sha256']==l['problem_sha256']
        assert hashlib.sha256(r['text'].encode()).hexdigest()==l['text_sha256']
        assert len(r['token_ids'])==r['tokens'] and r['thinking_tokens']==(r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids']))
        records.append(dict(l,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
    assert len(records)==1319
    refs['groups']['L27_L27']=dict(records=records,generation_seconds=data['generation_seconds'])
    refs['source_sha256'].update({str(prior/'L27_L27/result.json'):sha(prior/'L27_L27/result.json'),str(prior/'L27_L27/author_partial.jsonl'):hashlib.sha256(raw).hexdigest()})
    rel='trajectory_length_gsm_20260918_run1';out=HERE/rel;out.mkdir(exist_ok=False)
    plan=copy.deepcopy(oldplan);plan.update(run_id=rel,dataset_key='gsm8k',dataset='GSM8K',rows=priorplan['rows'],new_answers=1319,
        arms=[a for a in oldplan['arms'] if a['name']==name],hard_stop_seconds_per_arm=1200,process_hard_stop_seconds=1500,
        interpretation='Fixed MATH-selected direction transferred without tuning; exposed benchmark, not independent confirmation')
    release=copy.deepcopy(oldrelease)
    for n in oldrelease['artifact_sha256']:(out/n).write_bytes((parent/n).read_bytes())
    # plan and reference copies are superseded only inside a newly-created unpublished directory.
    (out/'plan.json').unlink();(out/'historical_compact.json').unlink()
    save(out/'plan.json',plan);save(out/'historical_compact.json',refs)
    save(out/'selection.json',dict(rule=oldplan['decision'],selected=name,decisions=decisions))
    release.update(artifact_root=rel,plan_relative_path=rel+'/plan.json',plan_sha256=sha(out/'plan.json'),
        engineering_parent_release_sha256=sha(parent/'release.json'),
        engineering_parent_relative_path=parent.name+'/release.json',
        artifact_sha256={n:sha(out/n) for n in release['artifact_sha256']})
    # Parent source hashes must remain identical except the explicitly allowed runner.
    release['source_sha256']['run_label_alignment.py']=hashlib.sha256((HERE/'run_label_alignment.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    save(out/'release.json',release)
    files={}
    with tarfile.open(root/'.codex_work/trajectory_length_vector_20260918_run2.tar.gz') as tf:
        for member in tf.getmembers():
            if member.isfile():files[member.name]=tf.extractfile(member).read()
    files['run_label_alignment.py']=(HERE/'run_label_alignment.py').read_bytes().replace(b'\r\n',b'\n')
    for p in out.iterdir():files[rel+'/'+p.name]=p.read_bytes()
    archive=root/('.codex_work/'+rel+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tf:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);tf.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(selected=name,archive=str(archive),sha256=sha(archive))))

if __name__=='__main__':main()
