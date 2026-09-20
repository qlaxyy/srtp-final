"""Author grading for all sixteen outputs; no statistical confirmation from8."""
import argparse,hashlib,json,sys,time,unicodedata
from pathlib import Path

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--rebalance',type=Path,required=True);p.add_argument('--selected-data',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.rebalance))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    started=time.monotonic();done=load(a.run/'completed.json');plan=load(a.run/'plan.json')
    dataset=a.selected_data;selected=[json.loads(x) for x in dataset.read_text(encoding='utf8').splitlines()]
    data={r['train_index']:r for r in selected};assert len(data)==len(selected)==8
    results={};arms={};grades={}
    for arm in plan['arms']:
        d=load(a.run/arm/'result.json');results[arm]=d;grades[arm]=[]
        assert len(d['records'])==8
        for r,q in zip(d['records'],plan['rows']):
            assert r['train_index']==q['train_index']
            row=data[r['train_index']];h=hashlib.sha256(''.join(unicodedata.normalize('NFKC',row['problem']).split()).encode()).hexdigest()
            assert h==r['problem_sha256']==q['problem_sha256']
            _,gold=parse_ground_truth(row,'math');answer=extract_answer(r['text'],'math')
            grades[arm].append(dict(train_index=r['train_index'],gold=gold,answer=answer,correct=bool(check_is_correct(answer,gold)),
                total_tokens=len(r['token_ids']),thinking_tokens=r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids']),
                capped=r['finish_reason']=='length',events=r['events']))
        g=grades[arm];arms[arm]=dict(correct=sum(r['correct'] for r in g),questions=8,
            mean_total_tokens=sum(r['total_tokens'] for r in g)/8,mean_thinking_tokens=sum(r['thinking_tokens'] for r in g)/8,
            caps=sum(r['capped'] for r in g),generation_seconds=d['generation_seconds'],extra_probe_forwards=0)
    old,new=[arms[x] for x in plan['arms']];og,ng=[grades[x] for x in plan['arms']]
    comparison=dict(total_delta_percent=100*(new['mean_total_tokens']/old['mean_total_tokens']-1),
        thinking_delta_percent=100*(new['mean_thinking_tokens']/old['mean_thinking_tokens']-1),
        wrong_to_right=[a['train_index'] for a,b in zip(ng,og) if a['correct'] and not b['correct']],
        right_to_wrong=[a['train_index'] for a,b in zip(ng,og) if not a['correct'] and b['correct']])
    passes=new['correct']>=old['correct'] and new['caps']<=old['caps'] and comparison['total_delta_percent']<0 and comparison['thinking_delta_percent']<0
    report=dict(arms=arms,comparison=comparison,passes_fixed_engineering_screen=passes,grades=grades,
        startup_seconds=done['startup_seconds'],wall_seconds=done['wall_seconds'],grading_seconds=time.monotonic()-started,
        timing_limit='Generation includes shared control instrumentation and output polling. No isolated lexical-kernel timing or saturation claim.',
        scope='Exposed8 engineering questions, seed42. Not independent confirmation, model-size generality or interaction evidence. No confidence interval on this selected sample.',
        decision='Eligible to design a separately fixed fresh-data screen; do not claim efficacy' if passes else 'Stop fixed phrase candidate; no retuning on8 and no full benchmark promotion',
        sha256={str(p):sha(p) for p in [dataset,a.rebalance/'utils/parser.py',a.rebalance/'utils/grader.py',a.run/'plan.json',*[a.run/x/'result.json' for x in plan['arms']]]})
    with (a.run/'grade.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('grades','sha256')},indent=2))

if __name__=='__main__':main()
