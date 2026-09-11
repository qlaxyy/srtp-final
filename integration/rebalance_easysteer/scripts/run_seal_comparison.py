"""Bounded self-calibrated SEAL comparison with one shared additive engine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from mechanism_candidates import ROOT,BASE,read,save,sha,require,read_vector


def validate_bundle(bundle):
    plan=read(bundle/'plan.json')
    require(plan['status']=='prepared_not_run' and plan['new_comparison_answers']==390,'Unplanned comparison scope')
    require(plan['runtime']['max_tokens']==16000 and plan['run_order']==['unsteered','original_dynamic','seal'],'Protocol changed')
    require(plan['runtime']==dict(max_tokens=16000,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,
        gpu_memory_utilization=.9,temperature=.7,top_p=.95,seed=42,async_scheduling=True,chunked_prefill=False,group_timeout_seconds=600),'Unexpected runtime fields')
    datasets={}
    for name,item in plan['datasets'].items():
        require(sha(bundle/item['file'])==item['sha256'],'Dataset changed')
        group=[json.loads(line) for line in (bundle/item['file']).read_text(encoding='utf-8').splitlines()]
        require(len(group)==item['count'] and all('problem' in r and 'answer' in r for r in group),'Missing rows')
        datasets[name]=group
    require({n:len(r) for n,r in datasets.items()}=={'engineering':8,'math_train':100,'aime2024':30},'Wrong counts')
    require(not any(plan['overlap_checks'].values()),'Overlapping comparison split')
    for name,digest in plan['source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Source changed: '+name)
    for name,item in plan['assets'].items():
        require(sha(bundle/item['vector'])==item['vector_sha256'],'Vector changed')
        require(sha(bundle/item['fit'])==item['fit_sha256'],'Metadata changed')
        read_vector(bundle/item['vector'],bundle/plan['assets']['original_dynamic']['vector'])
    return plan,datasets


def specs(plan,bundle,tokenizer):
    from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
    from easysteer.vectors import from_pt_direction
    boundary=sorted(i for t,i in tokenizer.get_vocab().items() if 'ĊĊ' in t)
    markers={name:tokenizer.encode(text,add_special_tokens=False) for name,text in [('think_start_token_id','<think>'),('think_end_token_id','</think>')]}
    require(all(len(ids)==1 for ids in markers.values()),'Non-single reasoning markers')
    common=dict(boundary_token_ids=boundary,**{name:ids[0] for name,ids in markers.items()})
    result={'unsteered':None}
    for name in ['original_dynamic','seal','seal_zero']:
        key='seal' if name.startswith('seal') else name;asset=plan['assets'][key]
        params=dict(common,**(plan['dynamic_parameters'] if key=='original_dynamic' else {'initial_coef':0. if name=='seal_zero' else 1.}))
        result[name]=SteeringSpec(vectors=[VectorSpec(
            data=from_pt_direction(str(bundle/asset['vector']),layers=[asset['decoder_output_layer']]),
            algorithm='rebalance' if key=='original_dynamic' else 'seal',layers=[asset['decoder_output_layer']],
            scale=1.,normalize=False,apply=ApplySpec(generation_tokens=boundary),params=params)])
    return result,boundary,common['think_end_token_id']


def runtime_check(plan,bundle,datasets):
    import torch
    import vllm
    from transformers import AutoTokenizer
    require(torch.cuda.is_available(),'GPU unavailable; stop without reinstall')
    require(Path(vllm.__file__).resolve()==(ROOT/'sources/EasySteer/vllm-steer/vllm/__init__.py').resolve(),'Different vLLM')
    for name,digest in plan['model_files_sha256'].items():require(sha(Path(plan['model'])/name)==digest,'Model asset changed')
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    created,boundary,end=specs(plan,bundle,tokenizer)
    from vllm.steer_vectors.api import to_engine_request
    from vllm.steer_vectors.rebalance import ReBalanceParams
    for i,name in enumerate(['original_dynamic','seal','seal_zero']):
        request=to_engine_request(created[name],name=name,int_id=i+1)
        params=ReBalanceParams.from_request(request)
        require(request.target_layers==([20] if name=='original_dynamic' else [19]),'Wrong actual layer')
        require(params.constant_control is (name!='original_dynamic'),'Wrong actual controller')
    sys.path.insert(0,str(ROOT/BASE/'eval'));from rebalance_static_eval import build_prompt
    max_prompt=max(len(tokenizer.encode(build_prompt(tokenizer,r['problem']))) for rows in datasets.values() for r in rows)
    require(max_prompt+16000<=32768,'Prompt overflow')
    actual={name:dict(algorithm=created[name].vectors[0].algorithm,layers=created[name].vectors[0].layers,params=created[name].vectors[0].params) for name in ['original_dynamic','seal','seal_zero']}
    return tokenizer,created,boundary,end,dict(status='runtime_spec_assets_checked_before_model',max_prompt_tokens=max_prompt,torch=torch.__version__,vllm=vllm.__version__,cuda=torch.version.cuda,resolved_specs=actual)


def grade(plan,bundle,out,datasets):
    ledger=read(out/'ledger.json');require(ledger['status']=='comparison_generated_grading_pending','Incomplete generation')
    require(ledger['plan_sha256']==sha(bundle/'plan.json'),'Different plan')
    require(not (out/'analysis.json').exists(),'Existing grading must be reused explicitly')
    sys.path.insert(0,str(ROOT/'sources/ReBalance'));from utils.parser import extract_answer,parse_ground_truth
    from utils.grader import check_is_correct
    analyses={};grader={n:sha(ROOT/'sources/ReBalance/utils'/n,source=True) for n in ['parser.py','grader.py']}
    for dataset in ['math_train','aime2024']:
        rows=datasets[dataset];groups={};metrics={}
        for arm in plan['run_order']:
            key=dataset+'_'+arm;raw=read(out/(key+'.json'));require(sha(out/(key+'.json'))==ledger['files_sha256'][key],'Raw changed')
            records=raw['records'];require(len(records)==len(rows),'Missing answers');scored=[];started=time.perf_counter()
            with (out/(key+'.author.partial.jsonl')).open('x',encoding='utf-8') as stream:
                for i,(row,record) in enumerate(zip(rows,records,strict=True)):
                    require(record['problem']==row['problem'] and record['gold']==row['answer'],'Pair mismatch')
                    _,gold=parse_ground_truth(row,'math');correct=bool(check_is_correct(extract_answer(record['text'],'math'),gold))
                    item=dict(index=i,correct=correct);scored.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
            scored_data=dict(status='completed',input_sha256=sha(out/(key+'.json')),records=scored,grader_sources=grader,seconds=time.perf_counter()-started)
            save(out/(key+'.author.json'),scored_data);groups[arm]=(records,scored)
            metrics[arm]=dict(count=len(rows),correct=sum(r['correct'] for r in scored),accuracy_percent=100*sum(r['correct'] for r in scored)/len(rows),
                mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/len(rows),mean_total_tokens=sum(r['tokens'] for r in records)/len(rows),
                capped=sum(r['finish_reason']=='length' or r['tokens']==16000 for r in records),generation_seconds=raw['summary']['generation_seconds'])
            print(dataset,arm,metrics[arm],flush=True)
        paired={}
        for control in ['unsteered','original_dynamic']:
            x,gx=groups[control];y,gy=groups['seal'];pairs=[]
            for i,(a,b,ga,gb) in enumerate(zip(x,y,gx,gy,strict=True)):
                pairs.append(dict(index=i,control_correct=ga['correct'],seal_correct=gb['correct'],thinking_token_delta=b['thinking_tokens']-a['thinking_tokens'],total_token_delta=b['tokens']-a['tokens']))
            paired[control]=dict(thinking_change_percent=100*(metrics['seal']['mean_thinking_tokens']/metrics[control]['mean_thinking_tokens']-1),
                total_change_percent=100*(metrics['seal']['mean_total_tokens']/metrics[control]['mean_total_tokens']-1),
                improved_indices=[r['index'] for r in pairs if not r['control_correct'] and r['seal_correct']],
                degraded_indices=[r['index'] for r in pairs if r['control_correct'] and not r['seal_correct']],per_question=pairs)
        analyses[dataset]=dict(groups=metrics,seal_vs=paired)
    save(out/'analysis.json',dict(status='completed',purpose='Bounded comparison, no selection or tuning on these benchmark results',datasets=analyses,plan_sha256=sha(bundle/'plan.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','comparison'],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--runtime-check',action='store_true');p.add_argument('--grade-only',action='store_true');a=p.parse_args()
    bundle=a.bundle.resolve();out=a.output.resolve();plan,datasets=validate_bundle(bundle)
    if a.grade_only:require(a.stage=='comparison','No efficacy grading of smoke');grade(plan,bundle,out,datasets);return
    if not a.execute and not a.runtime_check:
        print(json.dumps(dict(status='cpu_preflight_passed',stage=a.stage,outputs=24 if a.stage=='smoke' else 390,model_forwards=0)));return
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    require(sys.platform=='linux' and not out.exists(),'Fresh Linux output required')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty code')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU work')
    tokenizer,created,boundary,end,receipt=runtime_check(plan,bundle,datasets)
    if a.runtime_check:print(json.dumps(receipt));return
    if a.stage=='comparison':
        smoke=read(Path(plan['smoke_output'])/'ledger.json');require(smoke['status']=='smoke_passed_no_efficacy_claim' and smoke['plan_sha256']==sha(bundle/'plan.json'),'Smoke incomplete')
    import torch
    from vllm import LLM,SamplingParams
    sys.path.insert(0,str(ROOT/BASE/'eval'));from rebalance_static_eval import build_prompt,generate_records,summarize
    from runtime_guards import guard_dynamic_preemption
    out.mkdir(parents=True);ledger=dict(status='loading_model',stage=a.stage,started_unix=time.time(),plan_sha256=sha(bundle/'plan.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),runtime=receipt,files_sha256={},groups={})
    save(out/'ledger.json',ledger)
    try:
        started=time.perf_counter();llm=LLM(model=plan['model'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,
            gpu_memory_utilization=.9,enable_steer_vector=True,steer_algorithms=['rebalance','seal'],steer_graph_mode='in_graph',enforce_eager=False,
            enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42)
        ledger['startup_seconds']=time.perf_counter()-started;core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        state=runner.steer_vector_state;require(state.supports_kv_replay,'Missing historical replay support');preemptions=guard_dynamic_preemption(core.scheduler,state)
        names=['engineering'] if a.stage=='smoke' else ['math_train','aime2024'];order=['unsteered','seal_zero','seal'] if a.stage=='smoke' else plan['run_order'];cap=256 if a.stage=='smoke' else 16000
        for name in names:
            rows=datasets[name];prompts=[build_prompt(tokenizer,r['problem']) for r in rows];results={}
            for arm in order:
                key=name+'_'+arm;require(time.time()-ledger['started_unix']<plan['timeout_seconds'][a.stage],'Batch deadline')
                sample=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=cap,skip_special_tokens=True)
                before=dict(state.replay_counts);before_preemptions=preemptions['events']
                watchdog=threading.Timer(600,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
                try:records,seconds=generate_records(llm,prompts,rows,sample,set(boundary),created[arm],checkpoint_path=out/(key+'.partial.jsonl'))
                finally:watchdog.cancel()
                for record in records:
                    ids=record['token_ids'];require(record['tokens']==len(ids)<=cap and record['finish_reason'] in ['length','stop'],'Invalid completed output')
                    record['thinking_tokens']=ids.index(end) if end in ids else len(ids);record['thinking_ended']=end in ids;record['answer_tokens']=len(ids)-record['thinking_tokens']-int(end in ids)
                summary=summarize(records,seconds,cap);summary['preemptions']=preemptions['events']-before_preemptions
                summary['dynamic_kv_replay']={k:v-before[k] for k,v in state.replay_counts.items()}
                require(not state._suspended,'Unrestored suspended request state')
                # Worker retirement is delivered by the next scheduler batch.
                summary['states_pending_next_scheduler_cleanup']=len(state._dynamic_params)
                result=dict(status='completed',purpose='engineering_only' if a.stage=='smoke' else 'comparison',dataset=name,arm=arm,
                    plan_sha256=sha(bundle/'plan.json'),commit=ledger['commit'],dataset_sha256=plan['datasets'][name]['sha256'],max_tokens=cap,records=records,summary=summary)
                save(out/(key+'.json'),result);results[arm]=result;ledger['groups'][key]=summary;ledger['files_sha256'][key]=sha(out/(key+'.json'));save(out/'ledger.json',ledger)
                print(name,arm,'completed',len(records),'outputs',seconds,'generation seconds',flush=True)
                if arm=='seal_zero':require(all(x['token_ids']==y['token_ids'] for x,y in zip(results['unsteered']['records'],records,strict=True)),'Zero SEAL does not reproduce unsteered outputs')
        ledger.update(status='smoke_passed_no_efficacy_claim' if a.stage=='smoke' else 'comparison_generated_grading_pending',completed_unix=time.time(),gpu_work_finished=True)
    except BaseException as error:ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'ledger.json',ledger)
    print(json.dumps({k:v for k,v in ledger.items() if k!='groups'}))


if __name__=='__main__':main()
