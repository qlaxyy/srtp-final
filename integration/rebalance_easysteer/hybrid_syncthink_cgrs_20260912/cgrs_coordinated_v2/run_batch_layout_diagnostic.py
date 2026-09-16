"""Fixed-token and frozen-controller geometry comparisons; no effectiveness run."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from engineering import ROOT,HERE,read,save,sha
from prepare_screen import phash
from batch_layout_diagnostic import PHASES,FrozenPrefixTrace,ready_for_refill


def validate_plan(plan):
    assert plan['kind']=='frozen_prefix_layout_v1' and plan['run_id']=='cgrs_frozen_prefix_layout8_20260916'
    assert plan['phases']==list(PHASES)
    assert plan['runtime']==dict(dtype='bfloat16',max_tokens=128,max_model_len=17920,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,chunked_prefill=True,
        seed=42,temperature=.7,top_p=.95,ignore_eos=True)
    assert plan['assets']['decoder_output_layer']==21 and plan['assets']['model']=='DeepSeek-R1-Distill-Qwen-7B'
    assert [r['train_index'] for r in plan['rows']]==[1726,6746,7311,7123,3694,4550,5380,2667]
    for r in plan['rows']:
        assert phash(r['problem'])==r['problem_sha256'] and len(r['forced_token_prefix'])==128
        assert len(set(r['capture_positions']))==4 and all(0<=k<128 for k in r['capture_positions'])
        assert all(type(t) is int and 0<=t<151936 for t in r['prompt_token_ids']+r['forced_token_prefix'])
        assert r['forced_token_prefix']==r['RC14_token_prefix']
        k=r['first_difference_vs_RC14']
        assert r['RC14_token_prefix'][:k]==r['history_token_prefix'][:k]
        assert r['RC14_token_prefix'][k]!=r['history_token_prefix'][k]
    assert plan['max_engine_steps']=={'reference8':160,'repeat8':160,'single1':1100,'refill4plus4':200}
    assert plan['phase_seconds']=={'reference8':180,'repeat8':180,'single1':300,'refill4plus4':180}
    assert plan['process_seconds']==900


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-authorized',action='store_true');args=p.parse_args()
    if not args.gpu_authorized:raise ValueError('Explicit fixed-layout diagnostic authorization required')
    plan=read(args.plan);validate_plan(plan);receipt=read(args.receipt)
    assert receipt['gpu_authorized'] is True and receipt['scope']=='layout8_four_paths_128_only'
    assert receipt['plan_sha256']==sha(args.plan)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==receipt['execution_commit']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name,True)==digest,name
    for name,digest in plan['prerequisite_sha256'].items():assert sha(ROOT/name)==digest,name
    a=plan['assets']
    for name,meta in a['model_files'].items():assert sha(Path(a['model_path'])/name)==meta['sha256'],name
    for key in ('vector','fit'):assert sha(a[key]['path'])==a[key]['sha256'],key
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=args.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    save(out/'resolved_plan.json',plan);save(out/'execution_receipt.json',receipt)
    started=time.perf_counter();trace=None;core=None;reference=None;reports={}
    try:
        for name in ('test_native.py','test_sampler_diagnostic_native.py','test_batch_layout_native.py'):
            check=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,timeout=60)
            save(out/(name+'.json'),dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr));assert check.returncode==0,name
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['PYTHONNOUSERSITE']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        assert all(tok.encode(build_prompt(tok,r['problem']))==r['prompt_token_ids'] for r in plan['rows'])
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',data=from_pt_direction(a['vector']['path'],layers=[21]),algorithm='rebalance',scale=1.,layers=[21],normalize=False,
            apply=ApplySpec(generation_tokens=boundaries),params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=17920,max_num_seqs=32,max_num_batched_tokens=4096,
            gpu_memory_utilization=.94,enable_chunked_prefill=True,enable_prefix_caching=False,async_scheduling=False,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        save(out/'runtime.json',dict(vllm_path=vllm.__file__,torch=torch.__version__,startup_seconds=time.perf_counter()-started))
        for phase in PHASES:
            folder=out/phase;folder.mkdir();trace=FrozenPrefixTrace(llm,folder,plan['rows'],reference,phase=='repeat8')
            ids=[];external={};completed={};steps=0;began=time.perf_counter();pending=list(range(8));first=[]
            def enqueue(indices):
                new=llm.enqueue([dict(prompt_token_ids=plan['rows'][i]['prompt_token_ids']) for i in indices],
                    sampling_params=SamplingParams(seed=42,temperature=.7,top_p=.95,max_tokens=128,ignore_eos=True,skip_special_tokens=False),steering=steer,use_tqdm=False)
                trace.register(new,indices);ids.extend(new)
                ops=llm.llm_engine.output_processor.request_states
                external.update({ops[rid].external_req_id:rid for rid in new})
                return new
            with (folder/'partial.jsonl').open('x') as stream:
                if phase=='single1':enqueue([pending.pop(0)])
                elif phase=='refill4plus4':first=enqueue(pending[:4]);pending=pending[4:]
                else:enqueue(pending);pending=[]
                while pending or llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    if steps>=plan['max_engine_steps'][phase] or time.perf_counter()-began>plan['phase_seconds'][phase]:raise TimeoutError('Frozen diagnostic budget')
                    if phase=='refill4plus4' and pending and ready_for_refill(trace.counts,first):enqueue(pending);pending=[]
                    if phase=='single1' and pending and not llm.llm_engine.has_unfinished_requests() and not core.batch_queue:enqueue([pending.pop(0)])
                    for result in llm.llm_engine.step():
                        assert result.finished;rid=external[result.request_id];index=trace.requests[rid];tokens=list(result.outputs[0].token_ids)
                        assert tokens==plan['rows'][index]['forced_token_prefix'] and rid not in completed
                        item=dict(row=index,train_index=plan['rows'][index]['train_index'],token_ids=tokens,status='forced_engineering_not_graded')
                        completed[rid]=item;stream.write(json.dumps(item)+'\n');stream.flush()
                    steps+=1
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
            torch.cuda.synchronize();assert len(completed)==8
            reports[phase]=trace.finish();reports[phase].update(engine_steps=steps,generation_seconds=time.perf_counter()-began)
            save(folder/'result.json',reports[phase])
            if phase=='reference8':reference=dict(states=trace.states,logits=trace.logits,native_draws=trace.native_draws,max_probabilities=trace.max_probabilities,geometry=trace.geometry)
            trace.close();trace=None
        save(out/'summary.json',dict(status='complete_engineering_only',reports=reports,process_seconds=time.perf_counter()-started,
            caveat='Frozen RC14 tokens with newly captured reference R states, not recovered historical R states. Logits32 positions plus native draws/confidence1024 positions per phase. Geometry sensitivity is diagnostic, not effect size or throughput.'))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),seconds=time.perf_counter()-started,completed_phases=list(reports)))
        if trace is not None:
            import torch
            torch.save(dict(states=trace.states,logits=trace.logits,native_draws=trace.native_draws,max_probabilities=trace.max_probabilities),trace.folder/'incomplete_trace.pt')
            save(trace.folder/'incomplete_geometry.json',trace.geometry)
        if core is not None:save(out/'unfinished_prefixes.json',[dict(request_id=q.request_id,token_ids=list(q.output_token_ids)) for q in core.scheduler.requests.values()])
        raise
    finally:
        if trace is not None:trace.close()


if __name__=='__main__':main()
