"""User-requested RS-only full tests; reuse immutable U/R, never tune on tests."""
import argparse,json,subprocess
from pathlib import Path
from prepare_execution import phash,save,digest
import run_batch as rb

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE/'full_tests_20260913'


def local():
    DATA.mkdir(exist_ok=False)
    freeze=rb.read(ROOT/'integration/rebalance_easysteer/configs/final_results_20260909.json')
    entries=[];files={}
    for role,name,count in [('math_test','Math_Math500',500),('gsm8k_test','Math_GSM8K',1319)]:
        source=ROOT/'sources/ReBalance/Data'/name/'test.jsonl'
        rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==count
        with (DATA/(role+'.jsonl')).open('x',encoding='utf-8',newline='\n') as f:
            for i,r in enumerate(rows):
                item=dict(r,train_index=i,problem_sha256=phash(r['problem']),dataset=role)
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
                entries.append(dict(dataset=role,dataset_index=i,problem_sha256=item['problem_sha256'],purpose='RS_only_full_test_after_failed_confirmation_user_requested',status='reserved_for_fixed_test_no_parameter_selection'))
        files[role]=rb.sha(DATA/(role+'.jsonl'))
    save(DATA/'data_usage.json',dict(entries=entries,normalization='NFKC remove Unicode whitespace; SHA256 UTF-8',scope='Known frozen test questions reused only for new RS predictions at user request; U/R outputs reused read-only. These are benchmarks, not new training validation or an independent method selection pool.'))
    save(DATA/'plan.json',dict(run_id='s64_full_math500_gsm1319_RS_c256_run1_20260913',scope='RS only: MATH500 + GSM8K1319; 1819 new full answers, U/R historical reuse',authorization='User explicitly requested full MATH/GSM8K combination-only tests and original speed configuration',roles=['math_test','gsm8k_test'],counts=dict(math_test=500,gsm8k_test=1319),arms=['RS'],max_num_seqs=256,max_num_batched_tokens=32768,async_scheduling=True,max_tokens=16000,temperature=.7,top_p=.95,seed=42,method=dict(entropy_weight=.8,pacing_cap=64),files=files,whole_budget_seconds=2700,arm_budget_seconds=1200,estimated_minutes=[15,30],engineering='8 existing engineering questions x5 at512 cap, new256-row graph compatibility; no new full U/R/S tests',stop='Engineering mismatch, clock/nonfinite error, preemption/OOM, missing/duplicate answers, asset mismatch or wall budget. Preserve partial. No threshold tuning or automatic test rerun.',analysis='Report RS against frozen U and R, per-dataset paired bootstrap10000 seed20260912, correctness changes, thinking/total, caps, time and overhead. Historical timing is not a concurrent causal speed test. No standalone S on these tests, therefore no full factorial synergy claim. Preserve failed200-question confirmation conclusion; do not select parameters on these tests.',frozen_benchmarks=freeze['benchmarks'][:2]))
    print('LOCAL READY',files)


def remote():
    from transformers import AutoTokenizer
    from rebalance_static_eval import build_prompt
    import torch,vllm
    exp=rb.read(DATA/'plan.json');base=rb.read(HERE/'experiment_plan.json')
    for n,m in base['assets']['model_files'].items():assert rb.sha(Path(base['assets']['model_path'])/n)==m['sha256']
    for k in ('vector','fit'):assert rb.sha(base['assets'][k]['path'])==base['assets'][k]['sha256']
    rows={role:rb.dataset(role,True) for role in ['engineering']+exp['roles']}
    history={};folder=Path('/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908')
    for role,b in zip(exp['roles'],exp['frozen_benchmarks']):
        assert rb.sha(DATA/(role+'.jsonl'))==exp['files'][role]
        meta=b['artifacts'];ep=folder/meta['evaluation'];gp=folder/meta['grading']
        assert rb.sha(ep)==meta['evaluation_sha256'];assert rb.sha(gp)==meta['grading_sha256']
        evaluation=rb.read(ep);grades=rb.read(gp)
        assert grades['input_sha256']==rb.sha(ep)
        for key in ('parser.py','grader.py'):assert rb.sha(ROOT/'sources/ReBalance/utils'/key)==grades[key+'_sha256']
        protocol=evaluation['protocol']
        for key,value in dict(max_tokens=16000,max_model_len=32768,temperature=.7,top_p=.95,seed=42,easysteer_output_layer=20).items():assert protocol[key]==value,(key,protocol[key])
        assert evaluation['provenance']['vector_sha256']==base['assets']['vector']['sha256']
        assert evaluation['provenance']['calibration_fit_sha256']==base['assets']['fit']['sha256']
        groups={}
        for arm,key in [('U','baseline'),('R','rebalance_dynamic')]:
            records=evaluation[key]['records'];labels=grades['groups'][key]['records'];assert len(records)==len(labels)==exp['counts'][role]
            compact=[]
            for i,(x,g,row) in enumerate(zip(records,labels,rows[role])):
                assert x['dataset_index']==g['index']==i and x['problem']==row['problem'] and str(x['gold'])==str(row['answer'])
                ids=x['token_ids'];assert len(ids)==x['tokens']
                think=ids.index(151649) if 151649 in ids else len(ids);assert think==x['thinking_tokens']
                compact.append(dict(dataset_index=i,problem_sha256=row['problem_sha256'],correct=g['author_correct'],tokens=len(ids),thinking_tokens=think,capped=len(ids)==16000))
            assert sum(x['correct'] for x in compact)==b['groups'][key]['correct']
            groups[arm]=dict(records=compact,summary=b['groups'][key])
        history[role]=dict(groups=groups,evaluation_path=str(ep),evaluation_sha256=rb.sha(ep),grading_path=str(gp),grading_sha256=rb.sha(gp),protocol=protocol)
    save(ROOT/'historical_reference.json',history)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not apps.strip()
    tok=AutoTokenizer.from_pretrained(base['assets']['model_path'],local_files_only=True)
    prompts={role:[tok.encode(build_prompt(tok,r['problem'])) for r in rs] for role,rs in rows.items()}
    assert max(len(p) for ps in prompts.values() for p in ps)+16000<32768
    sources={}
    for folder in ('sources/EasySteer/vllm-steer/vllm','sources/EasySteer/easysteer','sources/ReBalance/utils','integration/rebalance_easysteer/eval','integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl'):sources[p.relative_to(ROOT).as_posix()]=rb.sha(p)
    assert rb.sha(ROOT/'baseline_model_runner.py')=='88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5'
    sources['baseline_model_runner.py']=rb.sha(ROOT/'baseline_model_runner.py')
    save(ROOT/'resolved_full.json',dict(expansion=exp,run_id=exp['run_id'],data_frozen=True,source_sha256=sources,plan_sha256=rb.sha(HERE/'experiment_plan.json'),assets=base['assets'],prompts=prompts,output='/root/autodl-tmp/results/easysteer/'+base['namespace']+'/'+exp['run_id'],environment=dict(build_tools=rb.check_build_tools(),torch=torch.__version__,vllm=vllm.__version__),deployment=rb.read(ROOT/'deployment_identity.json'),historical_reference_sha256=rb.sha(ROOT/'historical_reference.json')))
    print('READY',rb.sha(ROOT/'resolved_full.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote',action='store_true');a=p.parse_args()
    remote() if a.remote else local()
