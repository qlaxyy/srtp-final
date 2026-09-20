"""Native RC14 reproduction plus independent byte-verified KV-clone probes."""
import argparse, hashlib, importlib.util, json, math, os, signal, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def read(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--source-run',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args(); plan=read(args.plan); source=Path(args.source_run); out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',plan)
    save(out/'execution_amendment.json',dict(method='Live native reproduction and byte-verified disjoint KV clone, no chunked prefix replay',
        reason='Avoid known BF16 historical replay discrepancy without relaxing its gate',
        trajectory_forward_tokens=40814, main_outputs_used_only_for_exact_reference_check=True,
        max_probe_tokens=5568, wall_limit_seconds=600, source_sha256=sha(__file__)))
    start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s ceiling')));signal.alarm(600)
    results=[]; replay_seconds=probe_seconds=0.
    try:
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_STEER_EAGER_IN_GRAPH']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        # Legacy probe module imports a different policy.py. Keep that import isolated.
        oldpolicy=sys.modules['policy']; probe_dir=HERE.parent/'cgrs_probe_20260913'
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
        cp=load('policy',probe_dir/'policy.py'); cb=load('overlap_probe_backend',probe_dir/'backend.py');sys.modules['policy']=oldpolicy
        a=plan['assets']
        for name,meta in a['model_files'].items(): assert sha(Path(a['model_path'])/name)==meta['sha256']
        for k in ('vector','fit'): assert sha(a[k]['path'])==a[k]['sha256']
        native=read(source/'plan.json'); refs=read(source/'RC14_wsc_shadow/result.json')['records']
        tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',data=from_pt_direction(a['vector']['path'],layers=[20]),algorithm='rebalance',scale=1.,layers=[20],normalize=False,
            apply=ApplySpec(generation_tokens=boundaries),params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,max_num_seqs=32,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=True,compilation_config=0,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=False,seed=42)
        adapter=Adapter(llm,tok,mode='negative',gate_on=True); runner=adapter.runner; scheduler=adapter.core.scheduler
        base_sampler=adapter.original_sampler; rc_sampler=runner.sampler
        controls={r['train_index']:np.load(source/'RC14_wsc_shadow'/f"{r['train_index']}_control.npy") for r in refs}
        for r in refs: assert hashlib.sha256(controls[r['train_index']].tobytes()).hexdigest()==r['control_trace_sha256']
        mapping={}; checks={'tokens':0,'control_rows':0}
        class Gate:
            def __getattr__(self,n): return getattr(rc_sampler,n)
            def __call__(self,logits,batch,**kw):
                idx=batch.idx_mapping[:batch.num_reqs].long();s=runner.steer_vector_state
                pre=torch.stack([s._coefs[idx],s._prev_step_mean[idx],adapter.opening[idx].float(),adapter.thinking[idx].float(),adapter.count[idx].float()],1).cpu().numpy()
                ans=rc_sampler(logits,batch,**kw)
                post=torch.stack([adapter.opening[idx].float(),adapter.thinking[idx].float(),adapter.count[idx].float(),adapter.eligible_count[idx].float(),adapter.changed_count[idx].float()],1).cpu().numpy()
                tids=ans.sampled_token_ids[:,0].cpu().tolist()
                for j,rid in enumerate(batch.req_ids):
                    ref=mapping[rid];p=int(pre[j,4]);expected=controls[ref['train_index']][p]
                    assert tids[j]==ref['token_ids'][p], f"Native token drift {ref['train_index']}:{p}"
                    assert np.array_equal(np.concatenate([pre[j],post[j]]),expected,equal_nan=True),f"Native control drift {ref['train_index']}:{p}"
                    checks['tokens']+=1;checks['control_rows']+=1
                return ans
        gate=Gate();runner.sampler=gate
        backend=cb.Backend(llm,False,branch_mode='kv_clone',verify_cache=True)
        backend.tokenizer=tok;backend.deadline=start+590;backend.probe_batch_limit=8
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in native['rows']]
        ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=16000,skip_special_tokens=False),steering=steer,use_tqdm=False)
        mapping.update(zip(ids,refs)); cb.enable_cumulative(llm,ids)
        ops=llm.llm_engine.output_processor.request_states; external={ops[rid].external_req_id:rid for rid in ids}
        targets={(r['train_index'],r['accepted_generated_tokens']):r for r in plan['rows']};done=set();audit_done=False
        suffix=tok.encode(plan['probe']['prompt'],add_special_tokens=False); startup=time.monotonic()-start
        stream=(out/'probes.jsonl').open('x'); last_progress=start
        while len(done)<8:
            began=time.monotonic(); outputs=llm.llm_engine.step();replay_seconds+=time.monotonic()-began
            pending=[]
            for output in outputs:
                rid=external[output.request_id]; tokens=list(output.outputs[0].token_ids);idx=mapping[rid]['train_index']
                if output.finished:done.add(rid)
                row=targets.get((idx,len(tokens)))
                if row is not None:
                    assert not output.finished
                    prefix=prompts[ids.index(rid)]+tokens
                    assert len(prefix)==row['prefix_tokens']
                    assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==row['generated_prefix_sha256']
                    pending.append(dict(rid=rid,prefix=prefix,snapshot=backend.snapshot(rid,len(prefix)),row=row))
            if pending:
                began=time.monotonic()
                with cb.parked(scheduler) as held:
                    before=backend.primary_signature(held)
                    slots=[adapter.active[r.request_id] for r in held]
                    fields=[adapter.opening,adapter.thinking,adapter.count,adapter.eligible_count,adapter.changed_count,adapter.first_change]
                    ad_before=[f[slots].clone() for f in fields]
                    # Probe children bypass RC14's fresh-main admission and lexical sampler.
                    backend.original_add=adapter.original_add;runner.sampler.original=base_sampler
                    if not audit_done:
                        audits=[dict(rid=r.request_id,prefix=list(r.all_token_ids),snapshot=backend.snapshot(r.request_id,len(r.all_token_ids))) for r in held]
                        ar=backend.drain_probe_requests(audits,SamplingParams(temperature=0,max_tokens=1,skip_special_tokens=False),steer,[],audit=True)
                        for item,res in zip(audits,ar): backend.expected_logits[item['rid']]=res['logits']
                        audit_done=True
                    probed=backend.drain_probe_requests(pending,SamplingParams(temperature=0,max_tokens=32,seed=42,skip_special_tokens=False),steer,suffix)
                    backend.original_add=adapter.add_requests;runner.sampler.original=gate
                    assert before==backend.primary_signature(held),'Probe mutated parent state/cache'
                    assert all(torch.equal(x,f[slots]) for x,f in zip(ad_before,fields)),'Probe mutated RC14 state'
                probe_seconds+=time.monotonic()-began
                for item,res in zip(pending,probed):
                    secondary,status,chosen=cp.boxed_certainty(res['token_ids'],res['entropy'],tok,res['vocab_size'])
                    valid=status in ('complete','no_interior_tokens')
                    c=1-sum(res['entropy'])/len(res['entropy'])/math.log(res['vocab_size']) if valid else None
                    record=dict(item['row'],probe_token_ids=res['token_ids'],probe_text=tok.decode(res['token_ids'],skip_special_tokens=False),entropies=res['entropy'],vocab_size=res['vocab_size'],
                        cgrs_certainty=c,cgrs_high=None if c is None else c>.9,boxed_interior_certainty=secondary if status=='complete' else None,status=status)
                    results.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                save(out/'progress.json',dict(probes=len(results),checks=checks,elapsed=time.monotonic()-start))
            if time.monotonic()-last_progress>20:
                print(json.dumps(dict(probes=len(results),checks=checks,elapsed=time.monotonic()-start)),flush=True);last_progress=time.monotonic()
        stream.close(); assert len(results)==174 and checks['tokens']==40814
        assert len(backend.replay_checks)==8 and all(r['passed'] for r in backend.replay_checks)
        save(out/'complete.json',dict(status='complete',probes=len(results),checks=checks,startup_seconds=startup,trajectory_reproduction_seconds=replay_seconds,
            probe_and_preservation_seconds=probe_seconds,wall_seconds=time.monotonic()-start,
            generated_probe_tokens=sum(len(r['probe_token_ids']) for r in results),audit_tokens=8,forward_counts=backend.forward_counts,
            clone_bytes=backend.clone_bytes,clone_checks=backend.clone_checks,raw_logits_audit=backend.replay_checks,
            invalid=sum(r['cgrs_certainty'] is None for r in results),interpretation_allowed=sum(r['cgrs_certainty'] is None for r in results)<=.2*174))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),probes=len(results),wall_seconds=time.monotonic()-start));raise
    finally: signal.alarm(0)

if __name__=='__main__':main()
