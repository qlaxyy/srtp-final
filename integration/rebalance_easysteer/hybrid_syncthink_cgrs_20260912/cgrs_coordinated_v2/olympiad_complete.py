"""Complete only24 missing U outputs, then untouched R/RC675; preserve the timed-out run."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from engineering import ROOT,HERE,read,save


def sha(p,canonical=False):
    if canonical:return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate(plan):
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n,True)==h,n
    a=plan['assets']
    for n,m in a['model_files'].items():assert sha(Path(a['model_path'])/n)==m['sha256'],n
    for k in ('vector','fit'):assert sha(a[k]['path'])==a[k]['sha256'],k
    assert a['decoder_output_layer']==20
    assert plan['arms']==['U','R','RCnegative'] and len(plan['rows'])==675
    assert len({row['problem_sha256'] for row in plan['rows']})==675
    for row in plan['rows']:
        from prepare_screen import phash
        assert phash(row['problem'])==row['problem_sha256']
    assert read(plan['reconciliation_receipt'])['passed'] is True
    assert read(plan['reconciliation_receipt'])['rows_sha256']==sha(HERE/'olympiad675_run1/rows.json')
    assert read(plan['grader_receipt'])['passed'] is True
    assert read(plan['grader_receipt'])['plan_sha256']==sha(HERE/'olympiad675_run1/plan.json')
    assert plan['runtime']['max_tokens']==16000 and plan['runtime']['async_scheduling'] is False
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--phase',choices=['full'],required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--engineering-gate',type=Path);args=p.parse_args()
    if not args.gpu_authorized:raise ValueError('Explicit Olympiad675 three-arm GPU scope required')
    plan=read(args.plan);validate(plan)
    if args.phase=='full':
        gate=read(args.engineering_gate)
        expected_gate=plan.get('engineering_gate_plan_sha256') or (sha(args.plan) if plan.get('require_current_runtime_gate') else plan['parent_plan_sha256'])
        assert gate['passed'] and gate['plan_sha256']==expected_gate
        if plan.get('engineering_gate_plan_sha256'):
            tested=read(args.engineering_gate.parent/'resolved_plan.json')
            assert sha(plan['engineering_gate_plan_path'])==expected_gate
            assert tested==read(plan['engineering_gate_plan_path'])
            for key in ('runtime','assets','engineering_rows','rule'):assert tested[key]==plan[key],key
            for n,h in tested['source_sha256'].items():
                if n!=str(Path(__file__).relative_to(ROOT).as_posix()):assert h==plan['source_sha256'][n],n
    run_id=plan['engineering_run_id'] if args.phase=='engineering' else plan['run_id']
    out=args.output_root/run_id;out.mkdir(parents=True,exist_ok=False)
    save(out/'resolved_plan.json',plan)
    parent=plan['continuation']
    for source in parent.get('source_files',[]):assert sha(source['path'])==source['sha256']
    assert sha(parent['partial_path'])==parent['partial_sha256']
    assert sha(parent['gpu_csv_path'])==parent['gpu_csv_sha256']
    prior=[json.loads(line) for line in Path(parent['partial_path']).read_text().splitlines()]
    assert len(prior)==parent['reused_completed_outputs'] and len({x['dataset_index'] for x in prior})==len(prior)
    prior_ids={x['dataset_index'] for x in prior}
    assert sorted(set(range(675))-prior_ids)==parent['missing_dataset_indices']
    for rec in prior:
        row=plan['rows'][rec['dataset_index']]
        assert row['source_id']==rec['source_id'] and row['problem_sha256']==rec['problem_sha256']
        assert len(rec['token_ids'])==rec['tokens']<=16000
    started=time.perf_counter();results={};monitor=None
    try:
        check=subprocess.run([sys.executable,str(HERE/'test_native.py')],capture_output=True,text=True,timeout=60)
        save(out/'native_cpu_check.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
        assert check.returncode==0
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['PYTHONNOUSERSITE']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from replay_adapter import ReplayAdapter
        a=plan['assets'];r=plan['runtime'];tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        boundaries=sorted(i for piece,i in tok.get_vocab().items() if 'ĊĊ' in piece)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',
            data=from_pt_direction(a['vector']['path'],layers=[20]),algorithm='rebalance',scale=1.,layers=[20],normalize=False,
            apply=ApplySpec(generation_tokens=boundaries),params=dict(read(a['fit']['path'])['parameters'],
                boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=r['max_model_len'],
            max_num_seqs=r['max_num_seqs'],max_num_batched_tokens=r['max_num_batched_tokens'],
            gpu_memory_utilization=r['gpu_memory_utilization'],enable_chunked_prefill=True,
            enable_prefix_caching=False,async_scheduling=False,enable_steer_vector=True,
            steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        startup=time.perf_counter()-started
        save(out/'runtime_identity.json',dict(python=sys.version,torch=torch.__version__,vllm_path=vllm.__file__,
            runtime=r,plan_sha256=sha(args.plan),note='Sync replay, BF16 unchanged; no new shared controller'))
        cases=[(arm,plan['engineering_rows'],256) for arm in ('U','R','Roff','Rshadow','RCnegative')] if args.phase=='engineering' else [(arm,([row for row in plan['rows'] if row['dataset_index'] not in prior_ids] if arm=='U' else plan['rows']),16000) for arm in plan['arms']]
        for name,rows,cap in cases:
            folder=out/name;folder.mkdir(parents=True)
            mode='off' if name in ('U','R','Roff') else 'shadow' if name=='Rshadow' else 'negative'
            before=time.perf_counter();adapter=ReplayAdapter(llm,tok,mode=mode,gate_on=True)
            setup=time.perf_counter()-before
            state=runner.steer_vector_state;oldremove=state.remove_request;histories={}
            def capture(rid,manager):
                if rid in state._dynamic_indices:
                    idx=state._dynamic_indices[rid];histories[rid]=hashlib.sha256(state._history[idx,:state._history_lengths[rid]].cpu().numpy().tobytes()).hexdigest()
                return oldremove(rid,manager)
            if args.phase=='engineering':state.remove_request=capture
            replay_before=dict(state.replay_counts);generated={};io=0.;forced=False;steps=0
            preempt_events=[];oldpreempt=core.scheduler._preempt_request
            def track_preempt(req,now):
                preempt_events.append(dict(request_id=req.request_id,computed_tokens_discarded=req.num_computed_tokens))
                return oldpreempt(req,now)
            core.scheduler._preempt_request=track_preempt
            prompts=[tok.encode(build_prompt(tok,row['problem'])) for row in rows]
            assert max(map(len,prompts))+cap<r['max_model_len']
            gpu=(folder/'gpu.csv').open('x');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--loop-ms=1000'],stdout=gpu,stderr=subprocess.DEVNULL)
            torch.cuda.synchronize();began=time.perf_counter()
            try:
                ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=cap,skip_special_tokens=False),steering=None if name=='U' else steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states;mapping={ops[rid].external_req_id:rid for rid in ids};rowmap=dict(zip(ids,rows))
                save(folder/'request_mapping.json',{rid:dict(problem_sha256=row['problem_sha256'],dataset_index=row.get('dataset_index'),source_id=row.get('source_id',row.get('train_index'))) for rid,row in rowmap.items()})
                heartbeat=began
                with (folder/'partial.jsonl').open('x',encoding='utf8') as stream:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.perf_counter()-began>(parent.get('U_ceiling_seconds',900) if name=='U' else plan['arm_ceiling_seconds']):raise TimeoutError('Fixed Olympiad ceiling reached')
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];answer=output.outputs[0];ts=list(answer.token_ids)
                            rec=dict(rowmap[rid],request_id=rid,token_ids=ts,tokens=len(ts),thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),text=tok.decode(ts,skip_special_tokens=True),finish_reason=answer.finish_reason)
                            assert rid not in generated and len(ts)<=cap
                            generated[rid]=rec;t=time.perf_counter();stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();io+=time.perf_counter()-t
                        steps+=1
                        if time.perf_counter()-heartbeat>=30:
                            heartbeat=time.perf_counter()
                            save(folder/f'progress_{steps:07d}.json',dict(completed=len(generated),expected=len(rows),
                                elapsed_seconds=heartbeat-began,steps=steps,running=len(core.scheduler.running),
                                waiting=len(core.scheduler.waiting),preemptions=len(preempt_events),
                                discarded_computed_tokens=sum(e['computed_tokens_discarded'] for e in preempt_events),
                                completed_output_tokens=sum(x['tokens'] for x in generated.values())))
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
                records=[dict(generated[rid],R_history_sha256=histories.get(rid)) for rid in ids]
                results[name]=dict(status='complete',records=records,generation_seconds=seconds,setup_seconds=setup,
                    checkpoint_io_seconds=io,control_gpu_seconds=None,events=adapter.completed if adapter.enabled else {},
                    scheduler_preemptions=preempt_events,replay_counts=replay,replay_events=adapter.replay_events if adapter.enabled else [],forced_preemption=forced)
                if name=='U':
                    recovered=[dict(rec,request_id='prior::'+rec['request_id'],parent_request_id=rec['request_id'],R_history_sha256=None) for rec in prior]
                    merged=sorted(recovered+records,key=lambda x:x['dataset_index'])
                    assert [x['dataset_index'] for x in merged]==list(range(675))
                    assert len({x['request_id'] for x in merged})==675
                    results[name].update(records=merged,generation_seconds=None,new_generation_seconds=seconds,
                        generation_seconds_interval=[v+seconds for v in parent.get('prior_generation_seconds_interval',[2400,parent['failed_process_wall_seconds']])],
                        generation_time_note='Prior interrupted U attempts retained. Interval adds their recorded lower bounds and conservative whole-process upper bounds to this measured completion. Mixed runtime; do not compare as uninterrupted U speed.',
                        parent_gpu_csv_path=parent['gpu_csv_path'],parent_gpu_csv_sha256=parent['gpu_csv_sha256'],
                        continuation=dict(parent_partial_sha256=parent['partial_sha256'],reused_complete_outputs=len(prior),new_outputs=len(records),
                            lost_unfinished_output_tokens=None,prior_scheduler_preemptions=None,primary_R_RC_controls_continuous=True))
                save(folder/'result.json',results[name]);print(json.dumps(dict(phase=args.phase,name=name,completed=len(rows),seconds=seconds,replay=replay)),flush=True)
            finally:
                monitor.terminate();monitor.wait(timeout=10);monitor=None;gpu.close();state.remove_request=oldremove;core.scheduler._preempt_request=oldpreempt;adapter.close()
            if args.phase=='engineering':
                assert forced
                if name!='U':assert replay['restored']>=1
                if name in ('Roff','Rshadow'):
                    key=lambda d:[(r['token_ids'],r['R_history_sha256']) for r in d['records']]
                    assert key(results[name])==key(results['R']),name+' replay equivalence failed'
                if name=='Rshadow':assert sum(e['eligible'] for e in results[name]['events'].values())>0
                if name=='RCnegative':assert sum(e['changed'] for e in results[name]['events'].values())>0
        save(out/('engineering_gate.json' if args.phase=='engineering' else 'batch_status.json'),dict(passed=True,status='complete',plan_sha256=sha(args.plan),startup_seconds=startup,wall_seconds=time.perf_counter()-started))
    except BaseException as exc:
        if monitor:monitor.terminate();monitor.wait(timeout=10)
        if 'began' in locals():
            save(folder/'interrupted_arm.json',dict(generation_seconds_to_error=time.perf_counter()-began,
                completed=len(generated),scheduler_preemptions=preempt_events,
                unfinished=[dict(request_id=q.request_id,num_output_tokens=q.num_output_tokens)
                    for q in core.scheduler.requests.values()]))
        save(out/'failure.json',dict(error=repr(exc),completed=list(results),wall_seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
