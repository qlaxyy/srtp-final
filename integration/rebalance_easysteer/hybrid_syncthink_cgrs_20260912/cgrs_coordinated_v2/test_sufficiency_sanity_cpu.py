"""Parser and input isolation checks for fixed scalar sanity, no model imports."""
import json,hashlib
from pathlib import Path
from run_sufficiency_sanity import parse_score,input_ids

def main():
    from collections import UserDict
    for value in ([1,2],{'input_ids':[1,2]},UserDict(input_ids=[1,2])):
        assert input_ids(value)==[1,2]
    for value in ([[1,2]],{'input_ids':[[1,2]]},['1']):
        try:input_ids(value)
        except ValueError:pass
        else:raise AssertionError('Invalid token-ID layout accepted')
    for s,finish,expected in [('100','stop',100),(' 0\n','stop',0),('95','stop',95),
        ('100','length',None),('101','stop',None),('-1','stop',None),('100 because correct','stop',None),
        ('99.9','stop',None),('Confidence: 100','stop',None),('','stop',None),('1e2','stop',None)]:
        assert parse_score(s,finish)==expected
    plan=json.loads((Path(__file__).parent/'sufficiency_intake_20260917/plan.json').read_text(encoding='utf8'))
    assert len(plan['rows'])==16 and len({r['train_index'] for r in plan['rows']})==4
    assert sum(r['expected_stop'] for r in plan['rows'])==4
    for i in {r['train_index'] for r in plan['rows']}:
        assert {r['category'] for r in plan['rows'] if r['train_index']==i}=={'incomplete','incorrect_complete','correct_complete','mismatched'}
    for row in plan['rows']:
        content=plan['prompt']+'\n\nProblem:\n'+row['problem']+'\n\nProposed reasoning:\n'+row['thought']
        assert row['case_id'] not in content and 'expected_stop' not in content
        assert hashlib.sha256(row['thought'].encode()).hexdigest()==row['thought_sha256']
    assert plan['maximum_generated_tokens']==len(plan['rows'])*plan['max_tokens']==128
    print('PASS: 11 parser cases including cap rejection; 16 controlled inputs, labels not sent, thought hashes and budget checked. GPU/tokenizer not loaded.')

if __name__=='__main__':main()
