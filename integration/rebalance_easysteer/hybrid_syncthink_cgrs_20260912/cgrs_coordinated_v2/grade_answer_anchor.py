"""CPU author grading of existing boxed intermediate answers; no model."""
import argparse,hashlib,json,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--sha',required=True)
    p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha(a.input)==a.sha;data=json.loads(a.input.read_text());assert len(data['rows'])==500
    for name,h in data['grader_sha256'].items():assert sha(a.runtime_root/'sources/ReBalance/utils'/name)==h
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    a.output.mkdir(parents=True,exist_ok=False);rows=[];t=time.monotonic()
    with (a.output/'partial.jsonl').open('x') as f:
        for r in data['rows']:
            _,gold=parse_ground_truth(r['gold_row'],'math');cache={};bs=[]
            for b in r['boxes']:
                text=b['answer']
                if text not in cache:
                    pred=extract_answer('\\boxed{'+text+'}');cache[text]=dict(prediction=pred,correct=bool(check_is_correct(pred,gold)))
                bs.append(dict(b,**cache[text]))
            correct=[b for b in bs if b['correct']];anchor=correct[0]['token_stop'] if correct else None
            after=[s for s in r['repeats'] if anchor is not None and s['start']>=anchor]
            row=dict(question=r['question'],train_index=r['train_index'],problem_sha256=r['problem_sha256'],gold=gold,
                boxes=bs,first_correct_box_stop=anchor,thinking_tokens=r['thinking_tokens'],
                repeat_steps_after_correct_box=len(after),repeat_tokens_after_correct_box=sum(s['stop']-s['start'] for s in after))
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
    result=dict(input_sha256=a.sha,grader_sha256=data['grader_sha256'],seconds=time.monotonic()-t,rows=rows,
        questions_with_correct_box=sum(r['first_correct_box_stop'] is not None for r in rows),
        questions_with_repeat_after_correct_box=sum(r['repeat_steps_after_correct_box']>0 for r in rows),
        repeat_steps_after_correct_box=sum(r['repeat_steps_after_correct_box'] for r in rows),
        repeat_tokens_after_correct_box=sum(r['repeat_tokens_after_correct_box'] for r in rows),
        limitation='Correct boxed answer does not certify reasoning completed; exact repetition may still be useful. Offline training gold only; no inference gate or causal efficacy claim.')
    with (a.output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
