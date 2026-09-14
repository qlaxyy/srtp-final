"""Isolated orchestration for a fixed7B vocabulary ablation; native R/Sampler reused."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from engineering import ROOT, HERE, read, save
from run_7b import sha, validate as validate_assets
from prepare_screen import phash


def validate_receipt(plan, receipt, plan_path):
    assert receipt['gpu_authorized'] is True
    assert receipt['plan_sha256'] == sha(plan_path)
    if plan.get('candidate_kind')=='first_reflection_full':
        from history_full import validate_full_plan
        validate_full_plan(plan,receipt)
        return
    if plan.get('candidate_kind')=='first_reflection':
        assert plan['phase']=='engineering_only' and receipt['reuse_registered_engineering_rows'] is True
        assert plan['engineering_cap']==512 and len(plan['engineering_rows'])==8
        for row in plan['engineering_rows']:assert phash(row['problem'])==row['problem_sha256']
        assert plan['runtime']==dict(dtype='bfloat16',max_tokens=16000,max_model_len=17408,max_num_seqs=32,
            max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,
            chunked_prefill=True,seed=42,temperature=.7,top_p=.95)
        return
    assert receipt['data_reconciled'] is True and receipt['unregistered_claims_checked'] is True
    for side in ('local','remote'):
        item=receipt[side+'_reconciliation']
        assert sha(item['path'])==item['sha256']
        audit=read(item['path'])
        assert audit['passed'] and audit['rows_sha256']==plan['reserved_rows_sha256']
    primary='RChistory' if plan.get('candidate_kind')=='first_reflection_screen' else 'RC8'
    assert plan['arms']==['R','RC14',primary] and len(plan['rows'])==100
    expected=dict(dtype='bfloat16',max_tokens=16000,max_model_len=17408,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,
        chunked_prefill=True,seed=42,temperature=.7,top_p=.95)
    if primary=='RChistory' and 'context_amendment' in plan:
        amendment=plan['context_amendment']
        assert amendment['generated_answers_before_amendment']==0
        audit_path=ROOT/amendment['prompt_audit_path']
        assert sha(audit_path)==amendment['prompt_audit_sha256']
        audit=read(audit_path)
        assert [r['problem_sha256'] for r in audit['rows']]==[r['problem_sha256'] for r in plan['rows']]
        maximum=max(r['prompt_tokens'] for r in audit['rows'])
        expected['max_model_len']=((maximum+16000+511)//512)*512
    assert plan['runtime']==expected
    if primary=='RC8':
        assert plan['speed_profiles']=={'current32':{'max_num_seqs':32},'candidate48':{'max_num_seqs':48}}
    else:
        assert receipt['authorized_phases']==['screen']
    assert len({r['problem_sha256'] for r in plan['rows']})==100
    for i,row in enumerate(plan['rows']):
        assert row['dataset_index']==i and row['dataset']=='math_train'
        assert phash(row['problem'])==row['problem_sha256']


def cases(plan, phase):
    if plan.get('candidate_kind')=='first_reflection_full':
        assert phase=='full'
        return [(role,'negative','original14','after_first_reflection') for role in plan['datasets']]
    if plan.get('candidate_kind')=='first_reflection_screen':
        assert phase=='screen'
        return [('R','off','original14','none'),('RC14','negative','original14','none'),
                ('RChistory','negative','original14','after_first_reflection')]
    if plan.get('candidate_kind')=='first_reflection':
        assert phase=='engineering'
        return [('R','off','original14','none'),('Roff_history','off','original14','after_first_reflection'),
            ('RC14default','negative',None,'none'),('RC14explicit','negative','original14','none'),
            ('RChistory_shadow','shadow','original14','after_first_reflection'),
            ('RChistory','negative','original14','after_first_reflection')]
    if phase=='engineering':
        return [('R','off','original14'),('Roff8','off','narrow8'),
                ('RC14default','negative',None),('RC14explicit','negative','original14'),
                ('RC8shadow','shadow','narrow8'),('RC8','negative','narrow8')]
    if phase=='speed':return [('R','off','original14')]
    return [('R','off','original14'),('RC14','negative','original14'),('RC8','negative','narrow8')]


def validate_engineering_gate(plan, plan_path, gate_path):
    gate=read(gate_path)
    assert gate['passed']
    if plan.get('candidate_kind') in ('first_reflection_screen','first_reflection_full'):
        prior=plan['engineering_evidence']
        assert sha(gate_path)==prior['gate_sha256']
        assert gate['plan_sha256']==prior['plan_sha256']
        for name,digest in prior['unchanged_mechanism_sources'].items():
            assert sha(ROOT/name,True)==digest,name
    else:
        assert gate['plan_sha256']==sha(plan_path)


def validate_prompt_capacity(prompts, cap, model_len):
    maximum=max(map(len,prompts))
    assert maximum+cap<=model_len, (maximum,cap,model_len)
    return maximum


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--phase',choices=['engineering','screen','speed','full'],required=True)
    p.add_argument('--profile',choices=['current32','candidate48'])
    p.add_argument('--engineering-gate',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-authorized',action='store_true');args=p.parse_args()
    if not args.gpu_authorized:raise ValueError('Explicit fixed-batch GPU authorization required')
    plan=read(args.plan);receipt=read(args.receipt)
    if plan.get('candidate_kind')=='first_reflection':assert args.phase=='engineering'
    if plan.get('candidate_kind')=='first_reflection_screen':assert args.phase=='screen'
    if plan.get('candidate_kind')=='first_reflection_full':assert args.phase=='full'
    validate_receipt(plan,receipt,args.plan)
    assert args.phase in receipt['authorized_phases']
    validate_assets(plan)
    if args.phase in ('screen','full'):
        validate_engineering_gate(plan,args.plan,args.engineering_gate)
    runtime=dict(plan['runtime'])
    if args.phase=='speed':
        assert args.profile in receipt['authorized_speed_profiles']
        runtime.update(plan['speed_profiles'][args.profile])
    else:assert args.profile is None
    run_id=plan['run_ids'][args.phase]+('_'+args.profile if args.profile else '')
    out=args.output_root/run_id;out.mkdir(parents=True,exist_ok=False)
    save(out/'resolved_plan.json',plan);save(out/'execution_receipt.json',receipt)
    if args.phase=='full':
        from history_full import collect_references
        save(out/'historical_reference.json',collect_references(plan))
    started=time.perf_counter();results={};monitor=None;active=None
    try:
        tests=('test_native.py','test_narrow_native.py')
        if plan.get('candidate_kind') in ('first_reflection','first_reflection_screen','first_reflection_full'):tests+=('test_history_native.py',)
        for test in tests:
            check=subprocess.run([sys.executable,str(HERE/test)],capture_output=True,text=True,timeout=60)
            save(out/(test+'.json'),dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
            assert check.returncode==0,test
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        os.environ['PYTHONNOUSERSITE']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),
                      str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from replay_adapter import ReplayAdapter
        a=plan['assets'];tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        cap={'engineering':plan.get('engineering_cap',256),'screen':16000,'speed':1024,'full':16000}[args.phase]
        if args.phase=='full':
            prepared={role:[tok.encode(build_prompt(tok,r['problem'])) for r in sub['rows']] for role,sub in plan['datasets'].items()}
            maximum=max(validate_prompt_capacity(p,cap,runtime['max_model_len']) for p in prepared.values())
        else:
            rows=plan[{'engineering':'engineering_rows','screen':'rows','speed':'speed_rows'}[args.phase]]
            prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in rows]
            maximum=validate_prompt_capacity(prompts,cap,runtime['max_model_len'])
        save(out/'prompt_capacity.json',dict(max_prompt_tokens=maximum,max_model_len=runtime['max_model_len'],cap=cap))
        boundaries=sorted(i for piece,i in tok.get_vocab().items() if 'ĊĊ' in piece)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',
            data=from_pt_direction(a['vector']['path'],layers=[21]),algorithm='rebalance',
            scale=1.,layers=[21],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,
                        think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=runtime['max_model_len'],max_num_seqs=runtime['max_num_seqs'],
            max_num_batched_tokens=runtime['max_num_batched_tokens'],
            gpu_memory_utilization=runtime['gpu_memory_utilization'],enable_chunked_prefill=True,
            enable_prefix_caching=False,async_scheduling=False,enable_steer_vector=True,
            steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        save(out/'runtime_identity.json',dict(runtime=runtime,torch=torch.__version__,
            vllm_path=vllm.__file__,python=sys.version,startup_seconds=time.perf_counter()-started))
        ceiling=plan['arm_seconds'][args.phase]
        for case in cases(plan,args.phase):
            name,mode,profile=case[:3];history_gate=case[3] if len(case)==4 else 'none'
            if args.phase=='full':
                rows=plan['datasets'][name]['rows'];prompts=prepared[name]
            folder=out/name/plan['primary_candidate'] if args.phase=='full' else out/name
            folder.mkdir(parents=True);active=folder
            before=time.perf_counter();kw={} if profile is None else {'trigger_profile':profile}
            if history_gate!='none':kw['history_gate']=history_gate
            adapter=ReplayAdapter(llm,tok,mode=mode,gate_on=True,**kw)
            setup=time.perf_counter()-before
            state=runner.steer_vector_state;oldremove=state.remove_request;history={}
            def capture(rid,manager):
                if rid in state._dynamic_indices:
                    slot=state._dynamic_indices[rid]
                    history[rid]=hashlib.sha256(state._history[slot,:state._history_lengths[rid]].cpu().numpy().tobytes()).hexdigest()
                return oldremove(rid,manager)
            if args.phase=='engineering':state.remove_request=capture
            preempt=[];oldpreempt=core.scheduler._preempt_request
            def track(req,now):
                preempt.append(dict(request_id=req.request_id,computed_tokens_discarded=req.num_computed_tokens))
                return oldpreempt(req,now)
            core.scheduler._preempt_request=track
            replay_before=dict(state.replay_counts);generated={};steps=0;forced=False;io=0.
            gpu=(folder/'gpu.csv').open('x')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=gpu,stderr=subprocess.DEVNULL)
            torch.cuda.synchronize();began=time.perf_counter();heartbeat=began
            try:
                ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],
                    sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=cap,
                                                  skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states
                mapping={ops[rid].external_req_id:rid for rid in ids};rowmap=dict(zip(ids,rows))
                with (folder/'partial.jsonl').open('x',encoding='utf8') as stream:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.perf_counter()-began>ceiling:raise TimeoutError('Fixed arm ceiling; partials retained')
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];ans=output.outputs[0];ts=list(ans.token_ids)
                            assert rid not in generated and len(ts)<=cap
                            rec=dict(rowmap[rid],request_id=rid,token_ids=ts,tokens=len(ts),
                                thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),
                                text=tok.decode(ts,skip_special_tokens=True),finish_reason=ans.finish_reason)
                            generated[rid]=rec;t=time.perf_counter()
                            stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();io+=time.perf_counter()-t
                        steps+=1
                        if time.perf_counter()-heartbeat>=30:
                            heartbeat=time.perf_counter()
                            save(folder/f'progress_{steps:07d}.json',dict(completed=len(generated),expected=len(rows),
                                elapsed_seconds=heartbeat-began,running=len(core.scheduler.running),waiting=len(core.scheduler.waiting),
                                preemptions=len(preempt),discarded_computed_tokens=sum(e['computed_tokens_discarded'] for e in preempt)))
                        if args.phase=='engineering' and not forced and steps>=32:
                            req=core.scheduler.requests.get(ids[0])
                            if req is not None and req in core.scheduler.running and req.num_output_tokens>=16:
                                assert req.num_output_placeholders==0
                                core.scheduler.running.remove(req);core.scheduler._preempt_request(req,time.monotonic());forced=True
                    llm.llm_engine.step()
                    for rid in ids:
                        if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                torch.cuda.synchronize();seconds=time.perf_counter()-began
                assert len(generated)==len(rows)
                replay={k:state.replay_counts[k]-replay_before[k] for k in replay_before}
                result=dict(status='complete',records=[dict(generated[rid],R_history_sha256=history.get(rid)) for rid in ids],
                    generation_seconds=seconds,setup_seconds=setup,checkpoint_io_seconds=io,
                    engine_steps=steps,forced_preemption=forced,replay_counts=replay,scheduler_preemptions=preempt,
                    events=adapter.completed if adapter.enabled else {},
                    replay_events=adapter.replay_events if adapter.enabled else [],trigger_profile=profile or 'original14',
                    extra_probe_forwards=0,control_gpu_seconds=None)
                save(folder/'result.json',result);results[name]=result
                print(json.dumps(dict(arm=name,n=len(rows),seconds=seconds)),flush=True)
            except BaseException:
                save(folder/'interrupted_arm.json',dict(elapsed_seconds=time.perf_counter()-began,completed=len(generated),
                    scheduler_preemptions=preempt,unfinished=[dict(request_id=q.request_id,num_output_tokens=q.num_output_tokens)
                        for q in core.scheduler.requests.values()]))
                save(folder/'unfinished_prefixes.json',[
                    dict(request_id=q.request_id,token_ids=list(q.output_token_ids),
                         status='incomplete_not_graded') for q in core.scheduler.requests.values()])
                raise
            finally:
                monitor.terminate();monitor.wait(timeout=10);monitor=None;gpu.close()
                state.remove_request=oldremove;core.scheduler._preempt_request=oldpreempt;adapter.close()
            if args.phase=='engineering':
                assert forced and replay['restored']>=1
                key=lambda d:[(r['token_ids'],r['R_history_sha256']) for r in d['records']]
                if name in ('Roff8','RC8shadow','Roff_history','RChistory_shadow'):assert key(result)==key(results['R']),name
                if name=='RC14explicit':assert key(result)==key(results['RC14default'])
                if name=='RC8':assert sum(e['changed'] for e in result['events'].values())>0
                if name=='RChistory':
                    events=list(result['events'].values())
                    assert sum(e['changed'] for e in events)>0
                    assert all(e['first_change']<0 or 0<=e['first_reflection']<e['first_change'] for e in events)
                    assert any(a['token_ids']!=b['token_ids'] for a,b in zip(result['records'],results['RC14explicit']['records']))
        save(out/('engineering_gate.json' if args.phase=='engineering' else 'batch_status.json'),
            dict(passed=True,status='complete',phase=args.phase,plan_sha256=sha(args.plan),wall_seconds=time.perf_counter()-started))
    except BaseException as exc:
        if monitor:monitor.terminate();monitor.wait(timeout=10)
        save(out/'failure.json',dict(error=repr(exc),completed_arms=list(results),wall_seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
