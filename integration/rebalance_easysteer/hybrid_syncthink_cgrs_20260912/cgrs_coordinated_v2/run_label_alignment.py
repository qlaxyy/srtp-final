"""Bounded engineering/full runner for the fixed five-arm MATH500 ablation."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from prepare_label_alignment import sha,save

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def validate(plan,release,phase):
    gsm=plan.get('dataset_key')=='gsm8k'
    assert plan['seed']==42 and plan['max_new_tokens']==16000 and len(plan['rows'])==(1319 if gsm else 500)
    if plan.get('experiment_kind')=='margin_v1':
        assert [a['name'] for a in plan['arms']]==['MARGIN_L27']
        assert not plan['execution'].get('sync_replay',False)
    elif plan.get('experiment_kind')=='mti_v1':
        assert [a['name'] for a in plan['arms']]==['MTI_L27']
        assert not plan['execution'].get('sync_replay',False)
    elif plan.get('experiment_kind')=='harmonic_v1':
        assert [a['name'] for a in plan['arms']]==['HARMONIC_L27']
        assert not plan['execution'].get('sync_replay',False)
    elif plan.get('experiment_kind')=='type_split_v1':
        assert [a['name'] for a in plan['arms']] in (['CHECK7','SWITCH5'],['CHECK7'],['SWITCH5'])
        assert all(a['suppression_table'].startswith('type_split_20260917/') for a in plan['arms'])
    else:
        assert [a['name'] for a in plan['arms']]==(['L27_L27'] if gsm or plan.get('single_transfer') else ['L27_L27','T14_T14','T14_L27','CV_CV','L27_L27_control'])
    assert release['plan_sha256']==sha(HERE/release.get('plan_relative_path','label_alignment_20260917/plan_v2_math500.json'))
    assert release['table_schema']=='compact-token-classes-v1'
    assert release['source_hash_mode']=='lf-normalized'
    for n,h in release['source_sha256'].items():
        assert hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
    for n,h in release['artifact_sha256'].items():assert sha(HERE/release.get('artifact_root','label_alignment_20260917')/n)==h,n
    assert phase in ('engineering','full')


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['engineering','full'],required=True)
    p.add_argument('--engineering-result',type=Path)
    args=p.parse_args();release=read(args.release)
    plan=read(HERE/release.get('plan_relative_path','label_alignment_20260917/plan_v2_math500.json'));validate(plan,release,args.phase)
    if args.phase=='full':
        gate=read(args.engineering_result/'complete.json')
        assert gate['phase']=='engineering' and gate['passed']
        assert gate['release_sha256']==release.get('engineering_parent_release_sha256',sha(args.release))
        if 'engineering_parent_release_sha256' in release:
            if 'engineering_complete_sha256' in release:
                assert sha(args.engineering_result/'complete.json')==release['engineering_complete_sha256']
            parent_path=HERE/release.get('engineering_parent_relative_path','label_alignment_20260917/release_v2.json')
            prior=read(parent_path)
            assert sha(parent_path)==release['engineering_parent_release_sha256']
            for n,h in prior['source_sha256'].items():
                if n not in ('run_label_alignment.py','grade_label_alignment.py'):
                    assert release['source_sha256'][n]==h,n
            assert release['runtime_source_sha256']==prior['runtime_source_sha256']
            assert release['assets']==prior['assets']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'plan.json',plan);save(args.output/'release.json',release)
    began=time.monotonic();done=[];monitor=None
    def timeout(*_):raise TimeoutError('Fixed batch ceiling; preserve partial results')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200 if args.phase=='engineering' else plan.get('process_hard_stop_seconds',9000))
    try:
        assets=release['assets']
        for name,meta in assets['model_files'].items():assert sha(Path(assets['model_path'])/name)==meta['sha256'],name
        for key in ('fit','vector'):assert sha(assets[key]['path'])==assets[key]['sha256'],key
        for name,h in release['runtime_source_sha256'].items():
            import hashlib
            assert hashlib.sha256((args.runtime_root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,name
        sys.path[1:1]=[str(args.runtime_root/'sources/EasySteer/vllm-steer'),str(args.runtime_root/'sources/EasySteer'),
            str(args.runtime_root/'integration/rebalance_easysteer/eval')]
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        from label_alignment_adapter import AlignmentAdapter
        assert Path(vllm.__file__).resolve().is_relative_to(args.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        rt=plan.get('execution',{});sync_replay=rt.get('sync_replay',False)
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=rt.get('max_model_len',32768),
            max_num_seqs=rt.get('max_num_seqs',256),max_num_batched_tokens=rt.get('max_num_batched_tokens',32768),gpu_memory_utilization=rt.get('gpu_memory_utilization',.90),
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',
            enforce_eager=False,enable_chunked_prefill=rt.get('chunked_prefill',False),enable_prefix_caching=False,async_scheduling=not sync_replay,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        names=[a['name'] for a in plan['arms']]
        mti=plan.get('experiment_kind')=='mti_v1'
        margin=plan.get('experiment_kind')=='margin_v1'
        if args.phase=='engineering':names=(['L27_REFERENCE','L27_OFF','L27_SHADOW'] if mti or margin else ['RC14','RC14_extension_off'])+names
        rows=release['engineering_rows'] if args.phase=='engineering' else plan['rows']
        assert len(rows)==(8 if args.phase=='engineering' else len(plan['rows']))
        if args.phase=='full' and plan.get('recovery_indices') is not None:
            indices=plan['recovery_indices']
            assert mti and indices==sorted(set(indices)) and 0<len(indices)<len(rows)
            rows=[rows[i] for i in indices]
            assert [r['dataset_index'] for r in rows]==indices
        cap=(64 if mti else 512) if args.phase=='engineering' else 16000
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in rows]
        assert max(map(len,prompts))+cap<=rt.get('max_model_len',32768)
        save(args.output/'identity.json',dict(vllm=str(vllm.__file__),torch=torch.__version__,
            model=assets['model_path'],phase=args.phase,startup_seconds=time.monotonic()-began))
        reference=None
        for name in names:
            folder=args.output/name;folder.mkdir()
            calib='T14' if name in ('T14_T14','T14_L27') else 'CV' if name=='CV_CV' else None
            vp=HERE/'label_alignment_20260917'/calib/'auto_vector.pt' if calib else Path(assets['vector']['path'])
            fp=HERE/'label_alignment_20260917'/calib/'fit.json' if calib else Path(assets['fit']['path'])
            if name=='HARMONIC_L27':fp=HERE/'harmonic_confidence_20260918/fit.json'
            layer=assets['decoder_output_layer']
            assert read(fp)['decoder_output_layer']==layer
            steer=SteeringSpec(vectors=[VectorSpec(name='label_alignment_'+(calib or 'L27'),data=from_pt_direction(str(vp),layers=[layer]),
                algorithm='rebalance',scale=1.,layers=[layer],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
                params=dict(read(fp)['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
            setup_start=time.monotonic()
            arm=next((a for a in plan['arms'] if a['name']==name),{})
            if mti or margin:arm=plan['arms'][0]
            large=mti or margin or name in ('L27_L27','T14_L27') or 'suppression_table' in arm;lexical=name=='L27_L27_control'
            if large or lexical:
                table=HERE/arm['suppression_table'] if 'suppression_table' in arm else HERE/'label_alignment_20260917/tables'/('opening.npz' if large else 'search.npz')
                with np.load(table) as z:tables={k:z[k] for k in z.files}
                AdapterType=AlignmentAdapter
                if name=='HARMONIC_L27':
                    from harmonic_adapter import HarmonicAdapter
                    AdapterType=HarmonicAdapter
                if sync_replay:
                    from replay_label_alignment import ReplayAlignmentAdapter
                    AdapterType=ReplayAlignmentAdapter
                extra={}
                if margin and name!='L27_REFERENCE':
                    from reflection_margin_adapter import MarginAdapter
                    AdapterType=MarginAdapter
                    extra['margin_mode']='off' if name=='L27_OFF' else 'shadow' if name=='L27_SHADOW' else 'active'
                adapter=AdapterType(llm,tok,tables=tables,large_suppression=large,lexical_control=lexical,enabled=True,**extra)
            else:
                AdapterType=Adapter
                if sync_replay:
                    from replay_adapter import ReplayAdapter
                    AdapterType=ReplayAdapter
                adapter=AdapterType(llm,tok,mode='negative',gate_on=True)
            mti_adapter=None
            if mti and name!='L27_REFERENCE':
                from mti_native_adapter import MTIAdapter
                mti_adapter=MTIAdapter(adapter,tok,mode='off' if name=='L27_OFF' else 'shadow' if name=='L27_SHADOW' else 'active')
                if mti_adapter.mode!='off':
                    bs=mti_adapter.block_size
                    worst_blocks=((max(map(len,prompts))+cap+bs-1)//bs)*min(len(rows),runner.max_num_reqs)
                    available=mti_adapter.pool.get_num_free_blocks()
                    assert worst_blocks<=available, f'KV worst-case reservation unsafe: {worst_blocks}>{available}'
            if name=='RC14_extension_off':
                sampler=runner.sampler;unused=AlignmentAdapter(llm,tok,enabled=False)
                assert runner.sampler is sampler;unused.close();assert runner.sampler is sampler
                if plan.get('experiment_kind')=='harmonic_v1':
                    from harmonic_adapter import HarmonicAdapter
                    observe=runner.steer_vector_state.observe_sample
                    unused=HarmonicAdapter(llm,tok,enabled=False);unused.close()
                    assert runner.sampler is sampler and runner.steer_vector_state.observe_sample==observe
            setup=time.monotonic()-setup_start
            stream_gpu=(folder/'gpu.csv').open('x')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=stream_gpu,stderr=subprocess.DEVNULL)
            generated={};io=0.;forced=False;steps=0
            replay_before=dict(runner.steer_vector_state.replay_counts) if sync_replay else {}
            torch.cuda.synchronize();start=time.monotonic()
            try:
                ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=SamplingParams(
                    temperature=.7,top_p=.95,seed=42,max_tokens=cap,skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states
                mapping={ops[r].external_req_id:r for r in ids};rowmap=dict(zip(ids,rows))
                with (folder/'partial.jsonl').open('x',encoding='utf8') as f:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.monotonic()-start>(240 if args.phase=='engineering' else plan.get('hard_stop_seconds_per_arm',1500)):raise TimeoutError(name)
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];ans=output.outputs[0];ts=list(ans.token_ids)
                            assert rid not in generated and len(ts)<=cap
                            rec=dict(rowmap[rid],token_ids=ts,tokens=len(ts),thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),
                                text=tok.decode(ts,skip_special_tokens=True),finish_reason=ans.finish_reason)
                            generated[rid]=rec;ts0=time.monotonic();f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();io+=time.monotonic()-ts0
                        steps+=1
                        if sync_replay and args.phase=='engineering' and not forced and steps>=32:
                            req=core.scheduler.requests.get(ids[0])
                            if req is not None and req in core.scheduler.running and req.num_output_tokens>=16:
                                assert req.num_output_placeholders==0
                                core.scheduler.running.remove(req);core.scheduler._preempt_request(req,time.monotonic());forced=True
                    llm.llm_engine.step()
                    for rid in ids:
                        if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                torch.cuda.synchronize();seconds=time.monotonic()-start
                records=[generated[r] for r in ids];assert len(records)==len(rows)
                result=dict(status='complete',records=records,generation_seconds=seconds,setup_seconds=setup,checkpoint_io_seconds=io,
                    events=adapter.completed,extra_model_forward_count=0,probe_count=0,
                    replay_counts={k:runner.steer_vector_state.replay_counts[k]-v for k,v in replay_before.items()},
                    replay_events=getattr(adapter,'replay_events',[]),forced_preemption=forced,
                    control_gpu_seconds=None,timing_note='Generation includes adapter kernels and partial I/O; control overhead not independently isolated.')
                if mti_adapter is not None:
                    result['mti']=mti_adapter.report()
                    result['extra_model_forward_count']=mti_adapter.calls
                if sync_replay:
                    result['extra_model_forward_count']=None
                    result['timing_note']+=' KV replay is additional model work; restored request counts and replay-prefill token counts are recorded, forward calls not separately counted.'
                save(folder/'result.json',result)
                if args.phase=='engineering' and margin:
                    if name=='L27_REFERENCE':
                        reference=records;history=[adapter.completed[r]['R_history_sha256'] for r in ids]
                    elif name in ('L27_OFF','L27_SHADOW'):
                        assert all(a['token_ids']==b['token_ids'] for a,b in zip(reference,records)), 'Margin off/shadow changed tokens'
                        assert [adapter.completed[r]['R_history_sha256'] for r in ids]==history, 'Margin off/shadow changed history'
                    else:
                        assert sum(e['margin_changes'] for e in adapter.completed.values())>0,'No real margin intervention'
                        for base,rec,rid in zip(reference,records,ids):
                            first=adapter.completed[rid]['margin_first']
                            if first<0:first=len(rec['token_ids'])
                            assert base['token_ids'][:first]==rec['token_ids'][:first], 'Margin diverged before intervention'
                elif args.phase=='engineering' and mti:
                    if name=='L27_REFERENCE':
                        reference=records;history=[adapter.completed[r]['R_history_sha256'] for r in ids]
                    elif name in ('L27_OFF','L27_SHADOW'):
                        assert all(a['token_ids']==b['token_ids'] for a,b in zip(reference,records)), 'MTI off/shadow changed tokens'
                        assert [adapter.completed[r]['R_history_sha256'] for r in ids]==history, 'MTI off/shadow changed control history'
                        if name=='L27_SHADOW':assert mti_adapter.calls>0
                    else:
                        assert mti_adapter.calls>0
                        for base,rec,rid in zip(reference,records,ids):
                            first=mti_adapter.first.get(rid,len(rec['token_ids']))
                            assert base['token_ids'][:first]==rec['token_ids'][:first], 'MTI diverged before intervention'
                elif args.phase=='engineering':
                    if sync_replay:assert forced and result['replay_counts']['restored']>=1
                    if sync_replay and large:assert sum(e['changed'] for e in adapter.completed.values())>0
                    if name=='RC14':reference=records
                    elif name=='RC14_extension_off':
                        assert all(a['token_ids']==b['token_ids'] for a,b in zip(reference,records)), 'Disabled extension changed generation'
                        assert [adapter.completed[r]['R_history_sha256'] for r in ids]==history,'Disabled extension changed history'
                    else:
                        for base,rec in zip(reference,records):
                            first=next((i for i,t in enumerate(base['token_ids']) if t in boundaries),len(base['token_ids'])-1)
                            assert base['token_ids'][:first+1]==rec['token_ids'][:first+1], 'Divergence before first possible intervention'
                        if lexical:assert sum(e['lexical_control_changes'] for e in adapter.completed.values())>0
                        if name=='HARMONIC_L27':assert sum(e['harmonic_step_updates'] for e in adapter.completed.values())>0
                    if name=='RC14':history=[adapter.completed[r]['R_history_sha256'] for r in ids]
                done.append(name);print(json.dumps(dict(arm=name,count=len(records),seconds=seconds)),flush=True)
            finally:
                monitor.terminate();monitor.wait(timeout=10);monitor=None;stream_gpu.close()
                if mti_adapter is not None:mti_adapter.close()
                adapter.close()
        save(args.output/'complete.json',dict(phase=args.phase,passed=True,arms=done,
            release_sha256=sha(args.release),wall_seconds=time.monotonic()-began))
    except BaseException as e:
        save(args.output/'failure.json',dict(error=repr(e),completed_arms=done,wall_seconds=time.monotonic()-began));raise
    finally:
        if monitor is not None:monitor.terminate();monitor.wait(timeout=10)
        signal.alarm(0)


if __name__=='__main__':main()
