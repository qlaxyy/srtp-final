"""Protocol and historical references for fixed7B history-gated full evaluation."""
from pathlib import Path
from engineering import read,sha
from prepare_screen import phash


def validate_full_plan(plan,receipt):
    assert plan['phase']=='full_test' and plan['arms']==['RChistory']
    assert receipt['authorized_phases']==['full'] and receipt['exposed_test_reuse_acknowledged'] is True
    assert list(plan['datasets'])==['math_test','gsm8k_test']
    assert [len(s['rows']) for s in plan['datasets'].values()]==[500,1319]
    expected=dict(dtype='bfloat16',max_tokens=16000,max_model_len=17920,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,
        chunked_prefill=True,seed=42,temperature=.7,top_p=.95)
    if plan.get('model_family')=='1p5b':
        expected.update(max_model_len=32768,max_num_seqs=256,max_num_batched_tokens=32768,
            gpu_memory_utilization=.9,async_scheduling=True,chunked_prefill=False)
        assert plan['assets']['decoder_output_layer']==20
        assert len(plan['async_engineering_rows'])==8 and plan['async_engineering_cap']==512
        for row in plan['async_engineering_rows']:assert phash(row['problem'])==row['problem_sha256']
    assert plan['runtime']==expected
    hashes=set()
    for role,sub in plan['datasets'].items():
        for i,row in enumerate(sub['rows']):
            assert row['dataset_index']==i and row['split']=='test'
            assert phash(row['problem'])==row['problem_sha256'] and row['problem_sha256'] not in hashes
            hashes.add(row['problem_sha256'])


def collect_references(plan):
    if plan.get('model_family')=='1p5b':
        from full_eval import historical
    else:
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


def validate_1p5b_assets(plan):
    import subprocess
    from engineering import ROOT
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name,True)==digest,name
    a=plan['assets']
    assert a['model']=='DeepSeek-R1-Distill-Qwen-1.5B'
    for name,meta in a['model_files'].items():assert sha(Path(a['model_path'])/name)==meta['sha256'],name
    for key in ('vector','fit'):assert sha(a[key]['path'])==a[key]['sha256'],key
    assert a['decoder_output_layer']==read(a['fit']['path'])['decoder_output_layer']==20
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
