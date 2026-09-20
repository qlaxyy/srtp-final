"""CPU-only validation of complete native q75/q90 evaluation evidence."""
import argparse,hashlib,json,tarfile
from pathlib import Path
HOME=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=read(HOME/'rows.json');source=read(HOME/'summary.json')
    report={'archive_sha256':sha(a.archive),'groups':{},'new_training_answers':0,'new_training_forwards':0}
    labels_by_group={};saved={}
    with tarfile.open(a.archive) as t:
        for case in ['R','L27']:
            for quantile in ['q75','q90']:
                group=case+'_'+quantile
                prefix='native_q90_eval_20260920_run1/'+group+'_math500/'
                raw={n:t.extractfile(prefix+n).read() for n in ['complete.json','result.json','analysis.json','identity.json','author_labels.jsonl']}
                d=json.loads(raw['result.json']);complete=json.loads(raw['complete.json']);analysis=json.loads(raw['analysis.json'])
                labels=[json.loads(line) for line in raw['author_labels.jsonl'].splitlines()]
                assert complete['passed'] and not complete['engineering']
                assert complete['identity']['case']==case and complete['identity']['quantile']==quantile
                for key in ['fit_sha256','vector_sha256']:
                    assert complete['identity'][key]==source['groups'][case][quantile][key]
                assert len(d['records'])==len(labels)==500
                for i,(r,l,e) in enumerate(zip(d['records'],labels,rows)):
                    assert r['dataset_index']==l['dataset_index']==i
                    assert r['problem_sha256']==l['problem_sha256']==e['problem_sha256']
                    ids=r['token_ids'];assert len(ids)==r['tokens']==l['tokens']<=16000
                    n=ids.index(151649) if 151649 in ids else len(ids)
                    assert n==r['thinking_tokens']==l['thinking_tokens']
                summary={'count':500,'correct':sum(l['correct'] for l in labels),
                    'mean_tokens':sum(l['tokens'] for l in labels)/500,
                    'mean_thinking_tokens':sum(l['thinking_tokens'] for l in labels)/500,
                    'capped':sum(l['tokens']==16000 for l in labels)}
                for key,value in summary.items():assert analysis['summary'][key]==value
                labels_by_group[group]=labels
                result={'summary':summary,'generation_seconds':d['generation_seconds'],
                    'process_wall_seconds':complete['wall_seconds'],'comparisons':analysis['comparisons'],
                    'extra_model_forward_count':d['extra_model_forward_count'],'probe_count':d['probe_count']}
                if quantile=='q90':
                    base=labels_by_group[case+'_q75'];cmp=analysis['comparisons'][case+'_native_q75']
                    wc=[i for i,(x,y) in enumerate(zip(base,labels)) if not x['correct'] and y['correct']]
                    cw=[i for i,(x,y) in enumerate(zip(base,labels)) if x['correct'] and not y['correct']]
                    assert wc==cmp['wrong_to_right'] and cw==cmp['right_to_wrong']
                    result['matched_native_q75']=cmp
                    result['decision']='point_gate_pass_exploratory_only' if cmp['point_gate'] else 'stop_this_candidate'
                report['groups'][group]=result
                saved[group]={n:v for n,v in raw.items() if n!='result.json'}
    a.output.mkdir(parents=True,exist_ok=False)
    for group,files in saved.items():
        out=a.output/group;out.mkdir()
        for n,v in files.items():(out/n).write_bytes(v)
    (a.output/'verified_summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:{'summary':v['summary'],'seconds':v['generation_seconds'],
        'matched':v.get('matched_native_q75')} for k,v in report['groups'].items()},indent=2))
if __name__=='__main__':main()
