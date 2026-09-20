"""Two fixed calibration collections; no evaluation or vector fitting."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
from prepare import HOME,HERE,read,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sparse-gate',type=Path,required=True);p.add_argument('--failed-run',type=Path,required=True);a=p.parse_args()
    assert (a.failed_run/'failure.json').exists() and not (a.failed_run/'complete.json').exists()
    assert not (a.failed_run/'L27_batch448/result.json').exists()
    for source in ['R','L27']:
        for offset in range(0,500 if source=='R' else 448,64):assert (a.failed_run/(source+'_batch'+str(offset))/'result.json').exists()
    sparse=read(a.sparse_gate/'complete.json');assert sparse['passed'] and sparse['plan_sha256']==sha(HOME/'plan.json')
    gate=read(HERE/'online_endpoint_20260919/results_run3/complete.json');assert gate['passed']
    plan=read(HOME/'plan.json');rows=read(HOME/'rows.json');assets=plan['assets']
    assert len(rows)==500 and sha(HOME/'rows.json')==plan['source_rows_sha256']
    manifest=read(HOME/'manifest.json')
    for n,h in manifest.items():assert sha(HOME/n)==h,n
    for n,h in read(HOME/'dependencies.json').items():assert sha(HERE/n)==h,n
    for n,h in plan['runtime_hashes'].items():
        assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
    for n,v in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==v['sha256']
    for k in ['fit','vector']:assert sha(assets[k]['path'])==assets[k]['sha256']
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import shutil
    assert shutil.disk_usage(a.output.parent).free>=3*1024**3,'Need 3 GiB free before collection'
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def timeout(*_):raise TimeoutError('900 second last-batch recovery budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    results={}
    try:
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        sys.path[:0]=[str(HERE/'online_endpoint_20260919'),str(HERE),str(a.runtime_root/'sources/EasySteer/vllm-steer'),str(a.runtime_root/'sources/EasySteer')]
        import numpy as np
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from label_alignment_adapter import AlignmentAdapter
        from collector import ChunkObserver,ChunkAudit
        assert Path(vllm.__file__).resolve().is_relative_to(a.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        bounds=sorted(i for piece,i in tok.get_vocab().items() if 'ĊĊ' in piece)
        fit=read(assets['fit']['path']);assert fit['decoder_output_layer']==20
        steer=SteeringSpec(vectors=[VectorSpec(name='endpoint_parent',data=from_pt_direction(assets['vector']['path'],layers=[20]),
          algorithm='rebalance',scale=1.,layers=[20],normalize=False,apply=ApplySpec(generation_tokens=bounds),
          params=dict(fit['parameters'],boundary_token_ids=bounds,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
          max_num_seqs=64,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
          enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='split',
          enforce_eager=True,compilation_config=0,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        tables=dict(np.load(HERE/'iterative_recalibration_20260919/opening.npz'))
        all_rows=rows
        jobs=[('L27',448,all_rows[448:500])]
        for source,offset,rows in jobs:
            arm=source+'_batch'+str(offset)
            folder=a.output/arm;folder.mkdir();slots={};generated={}
            adapter=AlignmentAdapter(llm,tok,tables=tables,enabled=True,large_suppression=True) if arm.startswith('L27') else None
            audit=ChunkAudit(runner);observer=ChunkObserver(runner,bounds)
            old_add=runner.add_requests
            def add(output):
                old_add(output)
                for r in output.scheduled_new_reqs:
                    slot=runner.req_states.req_id_to_index[r.req_id]
                    assert slot not in slots.values(),'Slot reuse forbidden'
                    slots[r.req_id]=slot
            runner.add_requests=add;began=time.monotonic()
            try:
                ids=llm.enqueue([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows],
                  sampling_params=SamplingParams(temperature=0,top_p=1,seed=42,max_tokens=16000,skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states;mapping={ops[r].external_req_id:r for r in ids}
                with (folder/'partial.jsonl').open('x') as f:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];assert rid not in generated
                            generated[rid]=list(output.outputs[0].token_ids)
                            f.write(json.dumps(dict(request_id=rid,tokens=generated[rid]))+'\n');f.flush()
                    llm.llm_engine.step()
                torch.cuda.synchronize();seconds=time.monotonic()-began
                native=audit.export();captured=observer.export()
                record=[];changed=0
                for r,rid in zip(rows,ids):
                    n=native[slots[rid]];tokens=generated[rid]
                    assert n[:,0].astype(int).tolist()==tokens
                    assert np.isfinite(n[:,1]).all() and (n[:,1]>0).all() and (n[:,1]<=1).all()
                    with (folder/f'{r["dataset_index"]}_native.npy').open('xb') as f:np.save(f,n)
                    record.append(dict(problem_sha256=r['problem_sha256'],token_ids=tokens,native_sha256=hashlib.sha256(n.tobytes()).hexdigest()))
                    slot=slots[rid];meta=captured.metadata[slot];states=captured.hidden.get(slot,[]);plen=len(r['prompt_token_ids'])
                    assert [x[2] for x in meta]==tokens
                    assert [x[0] for x in meta]==list(range(plen-1,plen-1+len(tokens)))
                    assert [x[1] for x in meta[1:]]==tokens[:-1]
                    starts=[x[0]-plen for x in states]
                    pre=np.stack([x[1] for x in states]) if states else np.empty((0,1536),dtype=np.float32)
                    post=np.stack([x[2] for x in states]) if states else np.empty((0,1536),dtype=np.float32)
                    assert np.isfinite(pre).all() and np.isfinite(post).all()
                    changed+=int(np.any(pre!=post,axis=1).sum())
                    with (folder/f'{r["dataset_index"]}_hidden.npz').open('xb') as f:
                        np.savez(f,pre_hidden=pre,post_hidden=post,starts=np.array(starts),prompt_tokens=plen)
                results[arm]=dict(records=record,generation_seconds=seconds,changed_hidden_positions=changed)
                save(folder/'result.json',results[arm])
                assert shutil.disk_usage(a.output).free>=1024**3,'Stop before exhausting disk'
                # Step-first states need not be directly steered; engineering already proved coverage.
            finally:
                runner.add_requests=old_add;observer.close();audit.close()
                if adapter is not None:adapter.close()
        save(a.output/'complete.json',dict(passed=True,wall_seconds=time.monotonic()-start,scope='Recovery of last 52 L27 calibration questions only; not a continuous full run',failed_run=str(a.failed_run),plan_sha256=sha(HOME/'plan.json')))
    except BaseException as e:
        save(a.output/'failure.json',dict(error=repr(e),completed_arms=list(results),wall_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
