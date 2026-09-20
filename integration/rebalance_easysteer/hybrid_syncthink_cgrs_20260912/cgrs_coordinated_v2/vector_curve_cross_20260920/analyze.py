"""Validate crossed results and compute paired four-cell interactions."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
HOME=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    refs=read(HOME/'historical_compact.json');rows=read(HOME/'rows.json');expected=read(HOME/'summary.json')['groups']
    labels={'00':refs['groups']['L27_L27']['records'],'11':refs['groups']['new_vector_new_curve']['records']}
    report={'groups':{},'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest()};saved={}
    with tarfile.open(a.archive) as t:
        for arm,cell in [('old_vector_new_curve','01'),('new_vector_old_curve','10')]:
            prefix='vector_curve_cross_20260920_run1/'+arm+'/'
            files={n:t.extractfile(prefix+n).read() for n in ['result.json','analysis.json','complete.json','identity.json','author_labels.jsonl']}
            d=json.loads(files['result.json']);analysis=json.loads(files['analysis.json']);done=json.loads(files['complete.json'])
            lab=[json.loads(line) for line in files['author_labels.jsonl'].splitlines()]
            assert done['passed'] and done['rows']==500 and not done['engineering']
            assert done['identity']['arm']==arm
            for key in ['fit_sha256','vector_sha256']:assert done['identity'][key]==expected[arm][key]
            assert len(d['records'])==len(lab)==500
            for i,(r,l,e) in enumerate(zip(d['records'],lab,rows)):
                assert r['dataset_index']==l['dataset_index']==i
                assert r['problem_sha256']==l['problem_sha256']==e['problem_sha256']
                ids=r['token_ids'];assert r['tokens']==l['tokens']==len(ids)<=16000
                assert r['thinking_tokens']==l['thinking_tokens']==(ids.index(151649) if 151649 in ids else len(ids))
            calculated={'count':500,'correct':sum(l['correct'] for l in lab),'mean_tokens':np.mean([l['tokens'] for l in lab]),
                        'mean_thinking_tokens':np.mean([l['thinking_tokens'] for l in lab]),'capped':sum(l['tokens']==16000 for l in lab)}
            for k,v in calculated.items():assert v==analysis['summary'][k]
            for name,comparison in analysis['comparisons'].items():
                b=refs['groups'][name]['records']
                assert comparison['wrong_to_right']==[i for i,(x,y) in enumerate(zip(b,lab)) if not x['correct'] and y['correct']]
                assert comparison['right_to_wrong']==[i for i,(x,y) in enumerate(zip(b,lab)) if x['correct'] and not y['correct']]
            labels[cell]=lab;report['groups'][arm]={'analysis':analysis,'wall_seconds':done['wall_seconds']}
            saved[arm]={n:v for n,v in files.items() if n!='result.json'}
    for lab in labels.values():assert [l['problem_sha256'] for l in lab]==[r['problem_sha256'] for r in rows]
    rng=np.random.default_rng(20260920);report['factorial']={}
    for metric in ['tokens','thinking_tokens','correct']:
        x={c:np.array([l[metric] for l in lab],dtype=float) for c,lab in labels.items()}
        differences={'vector_at_old_curve':x['10']-x['00'],'curve_at_old_vector':x['01']-x['00'],
                     'vector_at_new_curve':x['11']-x['01'],'curve_at_new_vector':x['11']-x['10'],
                     'interaction':x['11']-x['10']-x['01']+x['00']}
        statistics={}
        for key,delta in differences.items():
            scale=100 if metric=='correct' else 1
            samples=[]
            for _ in range(100):
                idx=rng.integers(0,500,size=(200,500));samples.extend(delta[idx].mean(1)*scale)
            statistics[key]={'mean_difference':float(delta.mean()*scale),'ci95':np.quantile(samples,[.025,.975]).tolist()}
        report['factorial'][metric]=statistics
    a.output.mkdir(parents=True,exist_ok=False)
    for arm,files in saved.items():
        folder=a.output/arm;folder.mkdir()
        for n,v in files.items():(folder/n).write_bytes(v)
    (a.output/'verified_summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({'groups':{k:v['analysis']['summary'] for k,v in report['groups'].items()},'factorial':report['factorial']},indent=2))
if __name__=='__main__':main()
