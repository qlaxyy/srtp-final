"""Fixed two-arm phrase engineering batch. No early exit or WSC signal."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=read(a.plan);assert plan['arms']==['RC14','RC14_but_wait'] and len(plan['rows'])==8
    for name,digest in plan['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=a.output/plan['run_id'];out.mkdir(parents=True,exist_ok=False);save(out/'plan.json',plan)
    started=time.monotonic();results={}
    def timeout(*_):raise TimeoutError('900 second batch ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        assets=plan['assets']
        for name,meta in assets['model_files'].items():assert sha(Path(assets['model_path'])/name)==meta['sha256']
        for k in ('vector','fit'):assert sha(assets[k]['path'])==assets[k]['sha256']
        source=Path('/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917')
        for name,digest in plan['upstream_sha256'].items():
            # Prior manifest normalizes Windows checkout newlines for Python sources.
            raw=(source/name).read_bytes().replace(b'\r\n',b'\n')
            assert hashlib.sha256(raw).hexdigest()==digest,name
        sys.path[1:1]=[str(source/'sources/EasySteer/vllm-steer'),str(source/'sources/EasySteer'),str(source/'integration/rebalance_easysteer/eval')]
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1',VLLM_STEER_EAGER_IN_GRAPH='1')
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        from wsc_control_trace import ControlTrace
        assert Path(vllm.__file__).resolve().is_relative_to(source/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        boundaries=sorted(i for piece,i in tok.get_vocab().items() if 'ĊĊ' in piece)
        assert boundaries==plan['boundary_ids']
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',data=from_pt_direction(assets['vector']['path'],layers=[20]),
            algorithm='rebalance',scale=1.,layers=[20],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(assets['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=32,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',
            enforce_eager=True,compilation_config=0,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in plan['rows']]
        save(out/'prompts.json',prompts);startup=time.monotonic()-started
        for arm in plan['arms']:
            folder=out/arm;folder.mkdir();adapter=Adapter(llm,tok,mode='negative',gate_on=True,lexical_mode='but_wait' if arm.endswith('but_wait') else 'original')
            trace=ControlTrace(adapter,max_calls=16000);slots={};old_add=runner.add_requests
            def add(output):
                old_add(output)
                for r in output.scheduled_new_reqs:
                    slot=runner.req_states.req_id_to_index[r.req_id]
                    assert slot not in slots.values(),'No slot reuse in engineering batch'
                    slots[r.req_id]=slot
            runner.add_requests=add;began=time.monotonic();generated={}
            try:
                ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=16000,skip_special_tokens=False),steering=steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states;mapping={ops[rid].external_req_id:rid for rid in ids}
                with (folder/'partial.jsonl').open('x',encoding='utf8') as f:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.monotonic()-began>400:raise TimeoutError('400 second arm ceiling')
                        for output in llm.llm_engine.step():
                            assert output.finished
                            rid=mapping[output.request_id];ans=output.outputs[0]
                            generated[rid]=dict(token_ids=list(ans.token_ids),text=ans.text,finish_reason=ans.finish_reason)
                            f.write(json.dumps(dict(request_id=rid,**generated[rid]))+'\n');f.flush()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                torch.cuda.synchronize();seconds=time.monotonic()-began;controls=trace.export()
                records=[]
                for r,rid in zip(plan['rows'],ids):
                    c=controls[slots[rid]]
                    assert len(c)==len(generated[rid]['token_ids'])
                    with (folder/f"{r['train_index']}_control.npy").open('xb') as f:np.save(f,c)
                    records.append(dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],**generated[rid],events=adapter.completed[rid]))
                result=dict(records=records,generation_seconds=seconds);results[arm]=result;save(folder/'result.json',result)
                if arm=='RC14':
                    reference=read(plan['reference_result'])['records']
                    match=[r['train_index']==old['train_index'] and r['token_ids']==old['token_ids'] for r,old in zip(records,reference)]
                    save(folder/'reference_gate.json',dict(matches=match,scope='Old native full engineering RC14 reference; no benchmark repeated'))
                    assert len(match)==8 and all(match),'Native reference drift; do not run candidate'
                else:
                    audits=[]
                    for r in records:
                        i=r['train_index'];base=next(v for v in results['RC14']['records'] if v['train_index']==i)
                        first=r['events']['first_change'];c=np.load(folder/f'{i}_control.npy');old=np.load(out/'RC14'/f'{i}_control.npy')
                        end=len(r['token_ids']) if first<0 else first
                        ok=r['token_ids'][:end]==base['token_ids'][:end] and np.array_equal(c[:end+int(first>=0),:5],old[:end+int(first>=0),:5],equal_nan=True)
                        audits.append(dict(train_index=i,first_policy_difference=first,prefix_and_control_equal=ok))
                    save(folder/'prefix_gate.json',audits);assert all(r['prefix_and_control_equal'] for r in audits)
            finally:
                runner.add_requests=old_add;trace.close();adapter.close()
        save(out/'completed.json',dict(startup_seconds=startup,wall_seconds=time.monotonic()-started,arms=list(results),plan_sha256=sha(a.plan)))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),completed_arms=list(results),wall_seconds=time.monotonic()-started));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
