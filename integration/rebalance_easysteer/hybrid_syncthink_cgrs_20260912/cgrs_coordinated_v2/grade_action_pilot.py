"""Grade completed paired action labels with frozen author parser and grader."""
import argparse,hashlib,json,sys,time
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    plan=read(a.plan);assert plan['phase']=='action_labels';completion=read(a.result/'complete.json');assert completion['passed']
    for n,h in plan['grader_sha256'].items():assert sha(a.runtime/'sources/ReBalance/utils'/n)==h
    sys.path.insert(0,str(a.runtime/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    start=time.monotonic();arms={n:read(a.result/(n+'_complete.json')) for n in plan['arms']};rows=[]
    with (a.result/'grades_partial.jsonl').open('x',encoding='utf8') as partial:
        for i,source in enumerate(plan['rows']):
            _,gold=parse_ground_truth(source['gold_row'],'math');r={k:source[k] for k in ('train_index','problem_sha256')};r['arms']={}
            for name,records in arms.items():
                x=records[str(i)];assert x['train_index']==source['train_index']
                full=x['prefix_token_ids']+x['token_ids'];assert x['total_tokens']==len(full)<=16000
                assert x['thinking_tokens']==(full.index(151649) if 151649 in full else len(full))
                pred=extract_answer(x['full_text']);correct=bool(check_is_correct(pred,gold))
                r['arms'][name]=dict(correct=correct,prediction=pred,gold=gold,total=x['total_tokens'],thinking=x['thinking_tokens'],finish_reason=x['finish_reason'],closed=x['think_closed'])
            ap,sk=[r['arms'][n] for n in ('apply_a','skip')]
            complete=all(x['finish_reason']=='stop' and x['closed'] for x in (ap,sk))
            r['label']='inconclusive'
            if complete and ap['correct'] and sk['correct']:
                if ap['total']<sk['total'] and ap['thinking']<sk['thinking']:r['label']='apply_shorter_both_correct'
                elif sk['total']<ap['total'] and sk['thinking']<ap['thinking']:r['label']='skip_shorter_both_correct'
                elif ap['total']==sk['total'] and ap['thinking']==sk['thinking']:r['label']='tie_both_correct'
                else:r['label']='mixed_length_both_correct'
            elif complete and ap['correct']!=sk['correct']:r['label']='apply_only_correct' if ap['correct'] else 'skip_only_correct'
            rows.append(r);partial.write(json.dumps(r)+'\n');partial.flush()
    counts={k:sum(r['label']==k for r in rows) for k in sorted({r['label'] for r in rows})}
    result=dict(rows=rows,counts=counts,grading_seconds=time.monotonic()-start,plan_sha256=sha(a.plan),source_sha256={n:sha(a.result/(n+'_complete.json')) for n in arms},completion=completion,
        limitation='Training action-label feasibility only. One seed and early negative-action states. Not a vector improvement, benchmark comparison, or proof of step correctness.')
    with (a.result/'graded.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(counts=counts,grading_seconds=result['grading_seconds'])))
if __name__=='__main__':main()
