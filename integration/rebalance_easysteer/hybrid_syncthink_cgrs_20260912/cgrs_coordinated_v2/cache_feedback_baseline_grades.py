"""CPU-only cache of original training answer grades, while new generation runs."""
import argparse, hashlib, json, sys
from pathlib import Path
from grade_self_feedback import read, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    release=read(a.artifact/'release.json')
    for name in ['plan.json','baseline.json']:
        assert sha(a.artifact/name)==release['artifact_sha256'][name]
    for name,h in {'parser.py':'8d04ceb93e9ef50d61157fd1c8826a5cbf8d7f64cda85459f1166e19d8784208',
                   'grader.py':'0338f6549f2cb98806473e746baf40b00d9c5bd21a69bc7de0008e4ad4c8d768'}.items():
        assert sha(a.runtime_root/'sources/ReBalance/utils'/name)==h
    sys.path.insert(0,str(a.runtime_root/'sources/ReBalance'))
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    rows=read(a.artifact/'plan.json')['rows'];answers=read(a.artifact/'baseline.json')
    assert len(rows)==len(answers)==500
    with a.output.open('x',encoding='utf8') as f:
        for i,(row,r) in enumerate(zip(rows,answers)):
            assert row['problem_sha256']==r['problem_sha256'] and r['dataset_index']==i
            _,gold=parse_ground_truth(row,'math')
            label=dict(dataset_index=i,problem_sha256=row['problem_sha256'],text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                       correct=bool(check_is_correct(extract_answer(r['text']),gold)))
            f.write(json.dumps(label)+'\n');f.flush()


if __name__=='__main__':main()
