"""Verify full artifacts and apply the predeclared advancement rule locally."""
import argparse,csv,hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
from grade_label_alignment import compare,summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--root',default='mti_native_20260918_run4/full/')
    p.add_argument('--arm',default='MTI_L27',choices=['MTI_L27','MARGIN_L27'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root
    with tarfile.open(a.archive) as t:
        def raw(name):return t.extractfile(root+name).read()
        def read(name):return json.loads(raw(name))
        complete=read('complete.json');assert complete['passed'] and complete['phase']=='full'
        plan=read('plan.json');release=read('release.json');analysis=read('analysis.json')
        assert hashlib.sha256(raw('release.json')).hexdigest()==complete['release_sha256']
        assert hashlib.sha256(raw('plan.json')).hexdigest()==release['plan_sha256']
        data=read(a.arm+'/result.json');assert data['status']=='complete'
        labels=[json.loads(x) for x in raw(a.arm+'/author_partial.jsonl').decode().splitlines()]
        assert len(labels)==len(data['records'])==len(plan['rows'])==500
        candidate=[]
        for i,(r,label,q) in enumerate(zip(data['records'],labels,plan['rows'])):
            assert r['dataset_index']==label['dataset_index']==q['dataset_index']==i
            assert r['problem_sha256']==label['problem_sha256']==q['problem_sha256']
            assert hashlib.sha256(r['text'].encode()).hexdigest()==label['text_sha256']
            assert r['tokens']==len(r['token_ids'])<=16000
            thought=r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids'])
            assert r['thinking_tokens']==thought and type(label['correct']) is bool
            candidate.append(dict(label,tokens=r['tokens'],thinking_tokens=thought))
        refpath=Path(__file__).parent/release['artifact_root']/'historical_compact.json'
        assert hashlib.sha256(refpath.read_bytes()).hexdigest()==release['artifact_sha256']['historical_compact.json']
        refs=json.loads(refpath.read_text(encoding='utf8'))['groups']
        s=summary(candidate);comparisons={k:compare(v['records'],candidate) for k,v in refs.items()}
        saved=analysis['candidates'][a.arm]
        assert s==saved['summary'] and comparisons==saved['comparisons']
        assert saved['source_result_sha256']==hashlib.sha256(raw(a.arm+'/result.json')).hexdigest()
        assert saved['labels_sha256']==hashlib.sha256(raw(a.arm+'/author_partial.jsonl')).hexdigest()
        l27=summary(refs['L27_L27']['records'])
        eligible=(comparisons['L27_L27']['accuracy_observed_within_2pp']
            and comparisons['R']['accuracy_observed_within_2pp']
            and s['mean_tokens']<l27['mean_tokens'] and s['mean_thinking_tokens']<l27['mean_thinking_tokens']
            and s['capped']<=l27['capped'])
        recovered=complete.get('assembled_recovery',False)
        if recovered:
            provenance=read('recovery_provenance.json')
            assert provenance['coverage_complete'] and not provenance['uniform_schedule']
            assert provenance['source_by_dataset_index'].count('recovery32')==42
            utilization=[]
        else:
            provenance=None
            gpu=list(csv.reader(io.StringIO(raw(a.arm+'/gpu.csv').decode())))
            utilization=[float(row[1].strip()) for row in gpu if len(row)>=4]
        report=dict(status='Full500 identity/length/labels and paired statistics verified locally',
            archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),summary=s,
            comparisons=comparisons,advance_to_gsm_observed_rule=bool(eligible),
            eligible_without_uniform_schedule_recheck=bool(eligible and not recovered),
            recovery_provenance=provenance,
            generation_seconds=data['generation_seconds'],full_process_seconds=complete['wall_seconds'],
            grading_and_analysis_seconds=analysis['grading_and_analysis_seconds'],
            gpu_mean_utilization_percent=float(np.mean(utilization)) if utilization else None,mti=data.get('mti'),
            arm=a.arm,extra_model_forward_count=data.get('extra_model_forward_count'),
            margin_changes=sum(e.get('margin_changes',0) for e in data['events'].values()),
            margin_changed_questions=sum(e.get('margin_changes',0)>0 for e in data['events'].values()),
            limitations=['Repeatedly exposed test set; not independent confirmation.',
                'No standalone candidate-mechanism arm, so no additive synergy claim.']+
                ([
                'Branch host interval overlaps main-forward synchronization; not isolated additional GPU time.',
                'Invalid partial run3 is preserved and excluded from all metrics.'] if a.arm=='MTI_L27' else
                ['Historical controls; one seed; no claim that probability dominance proves useful reflection.'])+
                (['458 original plus42 recovered with changed concurrency; not a uniform-schedule confirmation.'] if recovered else []))
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('summary','generation_seconds','advance_to_gsm_observed_rule')},indent=2))


if __name__=='__main__':main()
