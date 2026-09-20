"""Use existing author grader; never load a model or generate an answer."""
import argparse,hashlib,json,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--input-sha256',required=True);ap.add_argument('--runtime-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.input)==a.input_sha256
    data=json.loads(a.input.read_text(encoding='utf8'));assert len(data['rows'])==166
    for n,h in data['grader_sha256'].items():assert sha(a.runtime_root/'sources/ReBalance/utils'/n)==h
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import parse_ground_truth,extract_answer
    from utils.grader import check_is_correct
    a.output.mkdir(parents=True,exist_ok=False);records=[];started=time.monotonic()
    with (a.output/'partial.jsonl').open('x',encoding='utf8') as f:
        for row in data['rows']:
            assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
            _,gold=parse_ground_truth(row['gold_row'],'math');pred=extract_answer(row['text'])
            result={k:row[k] for k in ('train_index','problem_sha256','text_sha256','under_steps','thinking_tokens')}
            result.update(gold=gold,prediction=pred,correct=bool(check_is_correct(pred,gold)))
            records.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
    result=dict(status='complete CPU-only calibration-parent answer audit',input_sha256=sha(a.input),
        grader_sha256=data['grader_sha256'],questions=len(records),correct_questions=sum(r['correct'] for r in records),
        under_steps=sum(r['under_steps'] for r in records),under_steps_in_correct_trajectories=sum(r['under_steps'] for r in records if r['correct']),
        seconds=time.monotonic()-started,records=records,
        limitations='Final answer correctness is not step validity or proof of sufficient exploration. These are selected calibration traces, not benchmark accuracy or independent evidence of compression.')
    with (a.output/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))

if __name__=='__main__':main()
