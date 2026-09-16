"""Bounded7B same-input engineering capture. Never imported by normal inference."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from engineering import ROOT,HERE,read,save,sha
from prepare_screen import phash
from sampler_diagnostic import capture_positions,SamplerDiagnostic


def validate_plan(plan):
    assert plan['kind']=='same_logits_engineering_v1' and plan['run_id']=='cgrs_same_logits8_20260916'
    assert plan['runtime']==dict(dtype='bfloat16',max_tokens=128,max_model_len=17920,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,chunked_prefill=True,
        seed=42,temperature=.7,top_p=.95,ignore_eos=True)
    assert plan['assets']['model']=='DeepSeek-R1-Distill-Qwen-7B' and plan['assets']['decoder_output_layer']==21
    assert [r['train_index'] for r in plan['rows']]==[1726,6746,7311,7123,3694,4550,5380,2667]
    assert len({r['problem_sha256'] for r in plan['rows']})==8
    for r in plan['rows']:
        assert phash(r['problem'])==r['problem_sha256']
        assert r['capture_positions']==capture_positions(r['first_difference_vs_RC14'])
    assert plan['max_engine_steps']==512 and plan['max_capture_bytes']==192*1024*1024
    assert plan['generation_seconds']==180 and plan['process_seconds']==600


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-authorized',action='store_true');args=p.parse_args()
    if not args.gpu_authorized:raise ValueError('New bounded GPU authorization required')
    plan=read(args.plan);validate_plan(plan);receipt=read(args.receipt)
    assert receipt['gpu_authorized'] is True and receipt['plan_sha256']==sha(args.plan)
    assert receipt['scope']=='same_logits8_128_only'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==receipt['execution_commit']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name,True)==digest,name
    a=plan['assets']
    for name,meta in a['model_files'].items():assert sha(Path(a['model_path'])/name)==meta['sha256'],name
    for key in ('vector','fit'):assert sha(a[key]['path'])==a[key]['sha256'],key
    assert read(a['fit']['path'])['decoder_output_layer']==21
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=args.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    save(out/'resolved_plan.json',plan);save(out/'execution_receipt.json',receipt)
    started=time.perf_counter();adapter=None;monitor=None;core=None;generated={}
    try:
        for name in ('test_native.py','test_history_native.py','test_sampler_diagnostic_native.py'):
            r=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,timeout=60)
            save(out/(name+'.json'),dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
            assert r.returncode==0,name
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['PYTHONNOUSERSITE']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in plan['rows']]
        assert max(map(len,prompts))+128<=17920
        save(out/'prompts.json',dict(token_ids=prompts,lengths=list(map(len,prompts))))
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',data=from_pt_direction(a['vector']['path'],layers=[21]),
            algorithm='rebalance',scale=1.,layers=[21],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=17920,max_num_seqs=32,
            max_num_batched_tokens=4096,gpu_memory_utilization=.94,enable_chunked_prefill=True,enable_prefix_caching=False,
            async_scheduling=False,enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        adapter=Adapter(llm,tok,mode='shadow',gate_on=True,history_gate='after_first_reflection')
        ids=llm.enqueue([dict(prompt_token_ids=p) for p in prompts],sampling_params=SamplingParams(
            seed=42,temperature=.7,top_p=.95,max_tokens=128,ignore_eos=True,skip_special_tokens=False),steering=steer,use_tqdm=False)
        ops=llm.llm_engine.output_processor.request_states;mapping={ops[rid].external_req_id:rid for rid in ids}
        targets={rid:set(row['capture_positions']) for rid,row in zip(ids,plan['rows'])}
        trace=SamplerDiagnostic(adapter,targets,out,plan['max_capture_bytes']);adapter.original_sampler=trace
        save(out/'runtime.json',dict(vllm_path=vllm.__file__,torch=torch.__version__,python=sys.version,
            startup_seconds=time.perf_counter()-started,request_mapping={rid:row['train_index'] for rid,row in zip(ids,plan['rows'])}))
        gpu=(out/'gpu.csv').open('x');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=gpu,stderr=subprocess.DEVNULL)
        began=time.perf_counter();steps=0
        with (out/'partial.jsonl').open('x',encoding='utf8') as stream:
            while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                if steps>=512 or time.perf_counter()-began>180:raise TimeoutError('Fixed engineering budget')
                for result in llm.llm_engine.step():
                    assert result.finished;rid=mapping[result.request_id];ts=list(result.outputs[0].token_ids)
                    assert rid not in generated and len(ts)==128
                    item=dict(request_id=rid,token_ids=ts,finish_reason=result.outputs[0].finish_reason,status='engineering_not_graded')
                    generated[rid]=item;stream.write(json.dumps(item)+'\n');stream.flush()
                steps+=1
            llm.llm_engine.step()
            for rid in ids:
                if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
        torch.cuda.synchronize()
        assert len(generated)==8 and trace.seen=={(rid,p) for rid,ps in targets.items() for p in ps}
        assert len(adapter.completed)==8
        assert all(e['changed']==0 for e in adapter.completed.values())
        save(out/'result.json',dict(passed=True,records=list(generated.values()),engine_steps=steps,
            capture_batches=trace.checks,capture_payload_bytes=trace.bytes,observations=len(trace.seen),
            generation_seconds=time.perf_counter()-began,process_seconds=time.perf_counter()-started,
            events=adapter.completed,additional_model_forwards_for_sampler_checks=0,
            interpretation='Same-input sampler gate only; new truncated R trajectory, not saved historical-prefix replay. Instrumented duration is not production throughput. No accuracy or compression evaluation.'))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),completed=len(generated),seconds=time.perf_counter()-started))
        if core is not None:save(out/'unfinished_prefixes.json',[dict(request_id=q.request_id,token_ids=list(q.output_token_ids)) for q in core.scheduler.requests.values()])
        raise
    finally:
        if monitor is not None:monitor.terminate();monitor.wait(timeout=10);gpu.close()
        if adapter is not None:
            if isinstance(adapter.original_sampler,SamplerDiagnostic):adapter.original_sampler=adapter.original_sampler.native
            adapter.close()


if __name__=='__main__':main()
