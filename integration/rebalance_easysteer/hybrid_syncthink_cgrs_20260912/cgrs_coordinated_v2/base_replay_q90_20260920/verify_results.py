"""Validate recovered complete outputs and extract paired decisions, CPU only."""
import argparse,json,hashlib,tarfile
from pathlib import Path
HOME=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=read(HOME/'rows.json');refs=read(HOME/'historical_compact.json')
    result={'scope':'Original replay q90; exposed full MATH500; historical q75 controls',
            'archive_sha256':sha(a.archive),'groups':{}}
    with tarfile.open(a.archive) as t:
        for case in ['R','L27']:
            prefix='base_replay_q90_20260920_run1/'+case+'_math500/'
            get=lambda n:json.load(t.extractfile(prefix+n))
            complete=get('complete.json');d=get('result.json');analysis=get('analysis.json')
            labels=[json.loads(line) for line in t.extractfile(prefix+'author_labels.jsonl')]
            assert complete['passed'] and not complete['engineering'] and len(labels)==len(d['records'])==500
            for i,(r,l,e) in enumerate(zip(d['records'],labels,rows)):
                assert r['problem_sha256']==l['problem_sha256']==e['problem_sha256']
                assert r['dataset_index']==l['dataset_index']==i
                assert r['tokens']==l['tokens']==len(r['token_ids'])<=16000
                ids=r['token_ids'];thinking=ids.index(151649) if 151649 in ids else len(ids)
                assert r['thinking_tokens']==l['thinking_tokens']==thinking
            summary=dict(count=500,correct=sum(l['correct'] for l in labels),
                         mean_tokens=sum(l['tokens'] for l in labels)/500,
                         mean_thinking_tokens=sum(l['thinking_tokens'] for l in labels)/500,
                         capped=sum(l['tokens']==16000 for l in labels))
            for k,v in summary.items():assert analysis['summary'][k]==v
            key=case+'_recalibrated_q75';base=refs['groups'][key]['records'];cmp=analysis['comparisons'][key]
            wc=[i for i,(x,y) in enumerate(zip(base,labels)) if not x['correct'] and y['correct']]
            cw=[i for i,(x,y) in enumerate(zip(base,labels)) if x['correct'] and not y['correct']]
            assert wc==cmp['wrong_to_right'] and cw==cmp['right_to_wrong']
            result['groups'][case]={'summary':summary,'generation_seconds':d['generation_seconds'],
                 'process_wall_seconds':complete['wall_seconds'],'matched_q75':cmp,
                 'all_comparisons':analysis['comparisons'],
                 'decision':'point_gate_pass_exploratory_only' if cmp['point_gate'] else 'stop_this_candidate',
                 'extra_model_forwards':d['extra_model_forward_count'],'probe_count':d['probe_count']}
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'verified_summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:{'summary':v['summary'],'generation_seconds':v['generation_seconds'],
        'matched_q75':v['matched_q75'],'decision':v['decision']} for k,v in result['groups'].items()},indent=2))

if __name__=='__main__':main()
