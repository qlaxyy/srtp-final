"""CPU verification of completed raw outputs and frozen paired references."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from prepare_label_alignment import sha,save
from grade_label_alignment import summary,compare

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=read(a.run/'plan.json');release=read(a.run/'release.json')
    assert read(a.run/'complete.json')['passed']
    assert sha(a.assets/'plan.json')==release['plan_sha256']
    assert read(a.assets/'plan.json')==plan
    refs=read(a.assets/release['references_artifact']);analysis=read(a.run/'analysis.json')
    assert sha(a.assets/release['references_artifact'])==release['artifact_sha256'][release['references_artifact']]
    results={}
    for arm in plan['arms']:
        name=arm['name'];folder=a.run/name;data=read(folder/'result.json')
        rows=data['records'];labels=[json.loads(s) for s in (folder/'author_partial.jsonl').read_text().splitlines()]
        assert len(rows)==len(labels)==len(plan['rows'])
        compact=[]
        for r,l,q in zip(rows,labels,plan['rows']):
            assert r['problem_sha256']==l['problem_sha256']==q['problem_sha256']
            assert r['dataset_index']==l['dataset_index']==q['dataset_index']
            assert hashlib.sha256(r['text'].encode()).hexdigest()==l['text_sha256']
            tokens=r['token_ids'];assert len(tokens)==r['tokens']<=16000
            assert r['thinking_tokens']==(tokens.index(151649) if 151649 in tokens else len(tokens))
            assert r['finish_reason'] in ['stop','length']
            compact.append(dict(l,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']))
        s=summary(compact);assert s==analysis['candidates'][name]['summary']
        comparisons={k:compare(g['records'],compact) for k,g in refs['groups'].items()}
        assert comparisons==analysis['candidates'][name]['comparisons']
        gpu=[]
        for row in csv.reader((folder/'gpu.csv').read_text().splitlines()):
            if len(row)==4:gpu.append([float(x) for x in row[1:]])
        gpu=np.array(gpu)
        events=list(data['events'].values())
        c=comparisons['L27_L27'];rc=comparisons['R']
        eligible=(c['tokens_percent']<0 and c['thinking_tokens_percent']<0 and
            s['capped']<=summary(refs['groups']['L27_L27']['records'])['capped'] and
            c['accuracy_delta_pp']>=-2 and rc['accuracy_delta_pp']>=-2)
        results[name]=dict(summary=s,comparisons=comparisons,generation_seconds=data['generation_seconds'],
            setup_seconds=data['setup_seconds'],eligible_gsm=eligible,
            gpu_samples=len(gpu),gpu_mean_percent=float(gpu[:,0].mean()),gpu_max_memory_mib=float(gpu[:,1].max()),
            events=sum(e['changed'] for e in events),extra_model_forward_count=data['extra_model_forward_count'],probe_count=data['probe_count'],
            limitation='Event count is implementation counter, not semantic reflection count; logit probability mass and isolated control-kernel time not measured.')
    eligible=[k for k,r in results.items() if r['eligible_gsm']]
    selected=min(eligible,key=lambda k:(results[k]['summary']['mean_tokens'],-results[k]['summary']['accuracy_percent'])) if eligible else None
    save(a.output,dict(status='Completed raw identity, length, labels integrity and paired statistics verified locally',
        candidates=results,selected_for_gsm=selected,
        raw_files={str(p.relative_to(a.run)):sha(p) for p in a.run.rglob('*') if p.is_file()},
        grading_seconds=analysis['grading_and_analysis_seconds'],
        limitation='Grading decisions are author-grader outputs, not a second independent semantic judge. Exposed test set, single generation seed, historical controls; no independent confirmation.'))
    print(json.dumps(dict(selected_for_gsm=selected,results={k:{'summary':v['summary'],'vs_L27':v['comparisons']['L27_L27'],'generation_seconds':v['generation_seconds'],'gpu_mean_percent':v['gpu_mean_percent']} for k,v in results.items()}),ensure_ascii=False))

if __name__=='__main__':main()
