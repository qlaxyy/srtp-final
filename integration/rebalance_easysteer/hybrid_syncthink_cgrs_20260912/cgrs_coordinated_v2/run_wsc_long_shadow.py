"""Single-arm native long observation; fixed prior512-token prefix gate; no intervention."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from engineering import ROOT, HERE, read, save, sha


def validate(plan):
    from prepare_screen import phash
    assert plan['phase']=='engineering_only' and plan['arms']==['RC14_wsc_shadow']
    assert plan['max_tokens']==16000 and plan['seed']==42
    assert [r['train_index'] for r in plan['rows']]==[647,1837,4972,1441,4432,3497,1483,229]
    for r in plan['rows']:
        assert r['split']=='train' and phash(r['problem'])==r['problem_sha256']
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name,True)==digest,name
    for name,digest in plan['wsc_asset_sha256'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in plan['reference_sha256'].items():assert sha(ROOT/name)==digest,name
    a=plan['assets']
    for name,meta in a['model_files'].items():assert sha(Path(a['model_path'])/name)==meta['sha256'],name
    for k in ('vector','fit'):assert sha(a[k]['path'])==a[k]['sha256'],k


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-authorized',action='store_true')
    args=p.parse_args()
    if not args.gpu_authorized:raise ValueError('Fresh bounded GPU authorization required')
    plan=read(args.plan);validate(plan)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=args.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',plan);start=time.monotonic();results={}
    def timeout(*_):raise TimeoutError('900 second process ceiling; preserve outputs')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        check=subprocess.run([sys.executable,str(HERE/'test_wsc_shadow_cpu.py')],capture_output=True,text=True,timeout=30)
        save(out/'native_cpu_check.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
        assert check.returncode==0,'Native CPU observer tests failed'
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        os.environ['PYTHONNOUSERSITE']='1'
        os.environ['VLLM_STEER_EAGER_IN_GRAPH']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        import vllm
        assert Path(vllm.__file__).resolve().is_relative_to((ROOT/'sources/EasySteer/vllm-steer').resolve()), 'Unexpected vLLM import'
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        from wsc_shadow import WSCShadow
        from wsc_control_trace import ControlTrace
        a=plan['assets'];tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        boundaries=sorted(i for piece,i in tok.get_vocab().items() if 'ĊĊ' in piece)
        decoded=sorted(i for i in tok.get_vocab().values() if i not in tok.all_special_ids and '\n\n' in tok.decode([i],skip_special_tokens=False))
        assert boundaries==decoded==plan['boundary_ids'], 'Tokenizer boundary mapping changed'
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',data=from_pt_direction(a['vector']['path'],layers=[20]),
            algorithm='rebalance',scale=1.,layers=[20],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=32,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',
            enforce_eager=True,compilation_config=0,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in plan['rows']]
        assert all(151648 in x and len(x)+16000<32768 for x in prompts)
        save(out/'tokenizer_boundaries.json',dict(ids=boundaries,rule="vocabulary piece contains double newline; RC14 boundary-compatible adaptation",tokenizer_sha256=sha(Path(a['model_path'])/'tokenizer.json')))
        startup=time.monotonic()-start
        for arm in plan['arms']:
            folder=out/arm;folder.mkdir()
            adapter=Adapter(llm,tok,mode='negative',gate_on=True)
            trace=ControlTrace(adapter,max_calls=16000)
            recorder=WSCShadow(runner,enabled=arm.endswith('shadow'),max_calls=16000)
            slots={};old_add=runner.add_requests;old_remove=runner._remove_request
            histories={}
            def remove(rid):
                if rid in adapter.active:
                    state=runner.steer_vector_state;slot=adapter.active[rid]
                    histories[rid]=state._history[slot,:state._history_lengths[rid]].detach().cpu().numpy().copy()
                return old_remove(rid)
            runner._remove_request=remove
            def add(output):
                old_add(output)
                for r in output.scheduled_new_reqs:
                    slot=runner.req_states.req_id_to_index[r.req_id]
                    assert slot not in slots.values(),'Slot reuse is outside engineering contract'
                    slots[r.req_id]=slot
            runner.add_requests=add
            generated={};began=time.monotonic();prefix_checked=False;prefix_seconds=0.0
            try:
                ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],
                    sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=16000,skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states
                mapping={ops[rid].external_req_id:rid for rid in ids}
                save(folder/'requests.json',{rid:dict(train_index=r['train_index'],problem_sha256=r['problem_sha256']) for rid,r in zip(ids,plan['rows'])})
                with (folder/'partial.jsonl').open('x',encoding='utf8') as f:
                    import json
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.monotonic()-began>600:raise TimeoutError('600 second arm ceiling')
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];answer=output.outputs[0]
                            assert rid not in generated
                            generated[rid]=dict(token_ids=list(answer.token_ids),finish_reason=answer.finish_reason)
                            f.write(json.dumps(dict(request_id=rid,**generated[rid]))+'\n');f.flush()
                        if len(recorder.frames)==512 and not prefix_checked:
                            gate_start=time.monotonic();cap=recorder.export();ct=trace.export()
                            reference=read(ROOT/plan['reference_result'])['records']
                            for row,rid,expected in zip(plan['rows'],ids,reference):
                                assert row['train_index']==expected['train_index']
                                assert [v['selected'] for v in cap[slots[rid]]]==expected['token_ids'], 'Prior512 token drift'
                                old=np.load(ROOT/plan['reference_controls'][str(row['train_index'])],allow_pickle=False)
                                assert np.array_equal(ct[slots[rid]],old,equal_nan=True), 'Prior512 control drift'
                            prefix_checked=True;prefix_seconds=time.monotonic()-gate_start
                            save(folder/'prefix_gate.json',dict(passed=True,questions=8,tokens_per_question=512,seconds=prefix_seconds))
                            del cap,ct
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                torch.cuda.synchronize();generation_seconds=time.monotonic()-began-prefix_seconds
                assert prefix_checked, 'Prior prefix gate not covered'
                control=trace.export()
                records=[dict(control_trace_sha256=hashlib.sha256(control[slots[rid]].tobytes()).hexdigest(),train_index=row['train_index'],problem_sha256=row['problem_sha256'],
                    **generated[rid],R_history_sha256=adapter.completed[rid]['R_history_sha256']) for row,rid in zip(plan['rows'],ids)]
                assert all(r['R_history_sha256'] is not None for r in records)
                results[arm]=dict(records=records,generation_seconds=generation_seconds,prefix_check_seconds=prefix_seconds,events=adapter.completed)
                for row,rid in zip(plan['rows'],ids):
                    history=histories[rid]
                    assert hashlib.sha256(history.tobytes()).hexdigest()==adapter.completed[rid]['R_history_sha256']
                    with (folder/(str(row['train_index'])+'_R_history.npy')).open('xb') as hf:np.save(hf,history)
                save(folder/'result.json',results[arm])
                for row,rid in zip(plan['rows'],ids):
                    with (folder/(str(row['train_index'])+'_control.npy')).open('xb') as f:np.save(f,control[slots[rid]])
                if recorder.enabled:
                    export_start=time.monotonic();capture=recorder.export()
                    for row,rid in zip(plan['rows'],ids):
                        seq=capture[slots[rid]]
                        assert [x['selected'] for x in seq]==generated[rid]['token_ids'],'Accepted-token alignment failure'
                        with (folder/(str(row['train_index'])+'.npz')).open('xb') as f:
                            np.savez(f,hidden=np.stack([x['hidden'] for x in seq]),input_ids=np.array([x['input_id'] for x in seq]),positions=np.array([x['position'] for x in seq]),selected=np.array([x['selected'] for x in seq]),prompt_tokens=np.array(len(prompts[plan['rows'].index(row)])))
                        plen=len(prompts[plan['rows'].index(row)])
                        assert seq[0]['position']==plen-1
                        assert [x['position'] for x in seq]==list(range(plen-1,plen-1+len(seq)))
                        assert [x['input_id'] for x in seq[1:]]==generated[rid]['token_ids'][:-1], 'Processed-token alignment failure'
                    save(folder/'export.json',dict(seconds=time.monotonic()-export_start,slots=slots))
            except BaseException:
                # Best-effort partial device evidence, including unfinished requests.
                # CUDA errors may make copying impossible; preserve that error too.
                try:
                    partial=recorder.export() if recorder.enabled else {}
                    for slot,seq in partial.items():
                        with (folder/f'partial_slot_{slot}.npz').open('xb') as f:
                            np.savez(f,positions=np.array([x['position'] for x in seq]),input_ids=np.array([x['input_id'] for x in seq]),selected=np.array([x['selected'] for x in seq]),hidden=np.stack([x['hidden'] for x in seq]))
                    save(folder/'partial_slots.json',slots)
                except BaseException as copy_error:
                    save(folder/'partial_export_failure.json',dict(error=repr(copy_error)))
                raise
            finally:
                runner.add_requests=old_add;runner._remove_request=old_remove;recorder.close();trace.close();adapter.close()
        assert sum(x['changed'] for x in results['RC14_wsc_shadow']['events'].values())>0,'No RC14 intervention coverage'
        save(out/'engineering_gate.json',dict(passed=True,plan_sha256=sha(out/'plan.json'),startup_seconds=startup,
            wall_seconds=time.monotonic()-start,scope='Long native shadow observation; first512 exactly matched old reference; later tokens have no paired baseline; no efficacy claim'))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),completed_arms=list(results),wall_seconds=time.monotonic()-start))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
