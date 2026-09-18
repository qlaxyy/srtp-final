"""Eight exposed training prompts, reference/capture only. No efficacy test."""
import argparse,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).parent;rel=read(a.release);plan=read(a.release.parent/'plan.json')
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600 second batch limit')));signal.alarm(600)
    try:
        assert sha(a.release.parent/'plan.json')==rel['plan_sha256']
        for n,h in rel['sources'].items():assert hashlib.sha256((here/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        assets=rel['assets'];runtime=Path(rel['runtime_root'])
        for n,h in rel['runtime_source_sha256'].items():assert hashlib.sha256((runtime/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        for n,d in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==d['sha256'],n
        for n in ('vector','fit'):assert sha(assets[n]['path'])==assets[n]['sha256'],n
        table=a.release.parent/'opening.npz';assert sha(table)==rel['table_sha256']
        sys.path[1:1]=[str(runtime/'sources/EasySteer/vllm-steer'),str(runtime/'sources/EasySteer'),str(runtime/'integration/rebalance_easysteer/eval')]
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from replay_label_alignment import ReplayAlignmentAdapter
        from counterfactual_observer import SnapshotObserver
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        assert Path(vllm.__file__).resolve().is_relative_to(runtime/'sources/EasySteer/vllm-steer')
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=16,max_num_batched_tokens=32768,gpu_memory_utilization=.9,enable_steer_vector=True,
            steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,enable_chunked_prefill=False,
            enable_prefix_caching=False,async_scheduling=False,seed=42)
        runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s);layer=assets['decoder_output_layer']
        steer=SteeringSpec(vectors=[VectorSpec(name='snapshot_L27',data=from_pt_direction(assets['vector']['path'],layers=[layer]),
            algorithm='rebalance',scale=1.,layers=[layer],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(assets['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        with np.load(table) as z:tables={k:z[k] for k in z.files}
        prompts=[dict(prompt_token_ids=tok.encode(build_prompt(tok,r['problem']))) for r in plan['rows']]
        all_results={};timings={};startup=time.monotonic()-start
        for arm in ('reference','observer'):
            cls=ReplayAlignmentAdapter if arm=='reference' else SnapshotObserver
            extra={} if arm=='reference' else dict(asset_identity=rel['identity'])
            adapter=cls(llm,tok,tables=tables,large_suppression=True,lexical_control=False,enabled=True,**extra)
            adapter.core.scheduler._preempt_request=adapter.reject_preempt
            torch.cuda.synchronize();began=time.monotonic()
            try:
                ids=llm.enqueue(prompts,sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=512,skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states;mapping={ops[r].external_req_id:(i,r) for i,r in enumerate(ids)};records={}
                with (a.output/(arm+'.jsonl')).open('x',encoding='utf8') as f:
                    while llm.llm_engine.has_unfinished_requests():
                        for result in llm.llm_engine.step():
                            assert result.finished
                            i,rid=mapping[result.request_id];answer=result.outputs[0]
                            rec=dict(train_index=plan['rows'][i]['train_index'],token_ids=list(answer.token_ids),control=adapter.completed[rid])
                            records[i]=rec;f.write(json.dumps(rec)+'\n');f.flush()
                    llm.llm_engine.step()
                assert len(records)==8;all_results[arm]=records
                if arm=='observer':
                    assert len(adapter.snapshots)==8,'Insufficient completed boundaries'
                    snapshots={plan['rows'][i]['train_index']:adapter.snapshots[rid] for i,rid in enumerate(ids)}
                    torch.save(snapshots,a.output/'snapshots.pt')
                    save(a.output/'capture_manifest.json',dict(snapshot_sha256=sha(a.output/'snapshots.pt'),rows=[dict(train_index=k,prefix_sha256=v['prefix_sha256'],generated_tokens=len(v['generated_ids'])) for k,v in snapshots.items()]))
            finally:
                torch.cuda.synchronize();timings[arm]=time.monotonic()-began;adapter.close()
        assert all_results['reference']==all_results['observer'],'Snapshot observer changed tokens or history'
        save(a.output/'complete.json',dict(passed=True,phase='capture_only',cases=8,short_outputs=16,new_token_cap=8192,
            model=assets['model_path'],release_sha256=sha(a.release),startup_seconds=startup,arm_seconds=timings,
            wall_seconds=time.monotonic()-start,fork_replay_validated=False,efficacy_test=False))
        print('PASS capture-only; fork replay still requires separate engineering',flush=True)
    except BaseException as exc:
        save(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
