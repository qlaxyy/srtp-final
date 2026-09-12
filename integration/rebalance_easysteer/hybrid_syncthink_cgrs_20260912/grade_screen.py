"""Exactly one author-grade pass on each completed screen output; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sources/ReBalance'))
from utils.parser import parse_ground_truth, extract_answer
from utils.grader import check_is_correct


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',default='screening',choices=['screening','math','gsm8k'])
    p.add_argument('--count',type=int,default=64)
    a=p.parse_args()
    data_dir='prepared_run1' if a.role=='screening' else 'expanded_20260913'
    rows=[json.loads(s) for s in (Path(__file__).parent/data_dir/(a.role+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    for arm in ('U','R','S','RS'):
        folder=a.output/a.role/arm
        target=folder/'author_grade.json'
        if target.exists():raise FileExistsError(target)
        raw=(folder/'result.json').read_bytes();result=json.loads(raw)
        assert result['status']=='complete' and len(result['records'])==len(rows)==a.count
        started=time.monotonic();graded=[]
        with (folder/'author_partial.jsonl').open('x',encoding='utf-8') as stream:
            for row,record in zip(rows,result['records']):
                assert row['train_index']==record['train_index'] and row['problem']==record['problem']
                _,gold=parse_ground_truth(row,'math')
                answer=extract_answer(record['text'],'math')
                correct=bool(check_is_correct(answer,gold))
                item=dict(train_index=row['train_index'],author_correct=correct,extracted_answer=answer)
                graded.append(item);stream.write(json.dumps(item,ensure_ascii=False)+'\n');stream.flush()
        with target.open('x',encoding='utf-8') as f:
            json.dump(dict(records=graded,correct=sum(x['author_correct'] for x in graded),
                           seconds=time.monotonic()-started,input_sha256=hashlib.sha256(raw).hexdigest(),
                           grader_sha256={name:hashlib.sha256((ROOT/'sources/ReBalance/utils'/name).read_bytes()).hexdigest()
                                          for name in ('parser.py','grader.py')}),f,ensure_ascii=False,indent=2)
        print('GRADED',arm,sum(x['author_correct'] for x in graded),flush=True)


if __name__=='__main__':main()
