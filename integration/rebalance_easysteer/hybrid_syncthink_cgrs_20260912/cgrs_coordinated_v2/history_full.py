"""Protocol and historical references for fixed7B history-gated full evaluation."""
from pathlib import Path
from engineering import read,sha
from prepare_screen import phash


def validate_full_plan(plan,receipt):
    assert plan['phase']=='full_test' and plan['arms']==['RChistory']
    assert receipt['authorized_phases']==['full'] and receipt['exposed_test_reuse_acknowledged'] is True
    assert list(plan['datasets'])==['math_test','gsm8k_test']
    assert [len(s['rows']) for s in plan['datasets'].values()]==[500,1319]
    assert plan['runtime']==dict(dtype='bfloat16',max_tokens=16000,max_model_len=17920,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,
        chunked_prefill=True,seed=42,temperature=.7,top_p=.95)
    hashes=set()
    for role,sub in plan['datasets'].items():
        for i,row in enumerate(sub['rows']):
            assert row['dataset_index']==i and row['split']=='test'
            assert phash(row['problem'])==row['problem_sha256'] and row['problem_sha256'] not in hashes
            hashes.add(row['problem_sha256'])


def collect_references(plan):
    from run_7b import historical
    result=historical(plan)
    ref=plan['rc14_reference'];analysis_path=Path(ref['analysis_path'])
    assert sha(analysis_path)==ref['analysis_sha256']
    old=read(analysis_path)
    for role,sub in plan['datasets'].items():
        item=ref['results'][role];p=Path(item['path']);assert sha(p)==item['sha256']
        data=read(p);summary=old['datasets'][role];records=data['records'];labels=summary['grades']
        assert data['status']=='complete' and len(records)==len(labels)==len(sub['rows'])
        compact=[]
        for i,(row,record,label) in enumerate(zip(sub['rows'],records,labels)):
            assert row['problem_sha256']==record['problem_sha256']==label['problem_sha256']
            assert record['dataset_index']==label['dataset_index']==i
            assert record['tokens']==len(record['token_ids'])
            compact.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=label['correct'],
                tokens=record['tokens'],thinking_tokens=record['thinking_tokens'],capped=record['finish_reason']=='length'))
        result[role]['groups']['RC14']=dict(records=compact,summary={k:v for k,v in summary.items() if k not in ('grades','comparisons')})
    return result
