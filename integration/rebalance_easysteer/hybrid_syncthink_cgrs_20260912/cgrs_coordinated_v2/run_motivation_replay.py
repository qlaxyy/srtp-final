"""Isolated saved-token confidence instrumentation; no lexical suppression."""
import argparse,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['engineering','full'],required=True)
    p.add_argument('--engineering-result',type=Path);p.add_argument('--forward-check',type=Path);args=p.parse_args()
    home=Path(__file__).parent;release=read(home/'release.json');assets=release['assets']
    for n,h in release['source_sha256'].items():assert sha(home/n)==h,n
    for n in ['rows','engineering']:assert sha(home/(n+'.json'))==release[n+'_sha256']
    for n,h in release['runtime_source_sha256'].items():
        assert hashlib.sha256((args.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
    for n,m in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==m['sha256'],n
    for n in ['fit','vector']:assert sha(assets[n]['path'])==assets[n]['sha256']
    if args.phase=='full':
        gate=read(args.engineering_result/'complete.json')
        assert gate['passed'] and gate['release_sha256']==sha(home/'release.json')
        assert read(args.forward_check)['passed']
    args.output.mkdir(exist_ok=False,parents=True)
    def timeout(*_):raise TimeoutError('Fixed replay time ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600 if args.phase=='engineering' else 1800)
    try:
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        sys.path[1:1]=[str(args.runtime_root/'sources/EasySteer/vllm-steer'),str(args.runtime_root/'sources/EasySteer'),str(args.runtime_root/'integration/rebalance_easysteer/eval')]
        import torch,numpy as np
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from motivation_replay_sampler import ReplayBuffers,MotivationReplaySampler
        started=time.monotonic();tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        boundaries=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=256,max_num_batched_tokens=32768,gpu_memory_utilization=.9,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',
            enforce_eager=False,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        def reject(*_):raise RuntimeError('Replay does not permit KV preemption')
        core.scheduler._preempt_request=reject
        fit=read(assets['fit']['path']);layer=assets['decoder_output_layer']
        steer=SteeringSpec(vectors=[VectorSpec(name='motivation_original_R',data=from_pt_direction(assets['vector']['path'],layers=[layer]),
            algorithm='rebalance',scale=1.,layers=[layer],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(fit['parameters'],boundary_token_ids=boundaries,think_start_token_id=151648,think_end_token_id=151649))])
        allrows=read(home/('engineering.json' if args.phase=='engineering' else 'rows.json'))
        summary=[]
        for group in ['U','R']:
            rows=allrows if args.phase=='engineering' else [r for r in allrows if r['group']==group]
            prompts=[r.get('prompt_token_ids') or tok.encode(build_prompt(tok,r['problem']),add_special_tokens=False) for r in rows]
            buffers=ReplayBuffers(runner.max_num_reqs,max(len(r['token_ids']) for r in rows),runner.device)
            native=runner.sampler;original_add=runner.add_requests;original_remove=runner._remove_request
            original_observe=runner.steer_vector_state.observe_sample
            pending={};completed={};active={};raw_checks=[]
            online_means=torch.full_like(buffers.logmax,float('nan'))
            def add(output):
                for r in output.scheduled_new_reqs:
                    assert r.req_id in pending and not r.num_computed_tokens
                    assert len(r.prefill_token_ids)==len(r.prompt_token_ids)
                original_add(output)
                for r in output.scheduled_new_reqs:
                    slot=runner.req_states.req_id_to_index[r.req_id];active[r.req_id]=slot
                    buffers.register(slot,pending[r.req_id]['token_ids'],len(r.prompt_token_ids))
                    online_means[slot].fill_(float('nan'))
            def remove(rid):
                if rid in active:
                    slot=active.pop(rid);completed[rid]=dict(logmax=buffers.completed(slot))
                    if group=='R':
                        # Only boundary means are exported; other positions can be NaN.
                        ids0=pending[rid]['token_ids'];stop=ids0.index(151649) if 151649 in ids0 else len(ids0)
                        positions=[j for j in range(stop) if ids0[j] in boundaries]
                        means=online_means[slot,positions].cpu().tolist()
                        first=0;checks=[]
                        for j,value in zip(positions,means):
                            if j>first:
                                expected=float(np.exp(np.asarray(completed[rid]['logmax'][first:j],dtype=np.float64)).mean())
                                assert abs(value-expected)<2e-6,(rid,j,value,expected)
                                checks.append(dict(position=j,online_mean=value,offline_mean=expected))
                            first=j+1
                        completed[rid]['step_checks']=checks
                return original_remove(rid)
            def observe(batch,tokens,probabilities,*rest):
                idx=batch.idx_mapping[:batch.num_reqs].long()
                valid=torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=runner.device)
                pos=(buffers.count[idx]-1).clamp(min=0)
                expected=buffers.logmax[idx,pos].exp()
                torch._assert_async((~valid | ((expected-probabilities).abs()<1e-7)).all(),'Native raw probability mismatch')
                raw_checks.append(1)
                result=original_observe(batch,tokens,probabilities,*rest)
                online_means[idx,pos]=torch.where(valid,runner.steer_vector_state._prev_step_mean[idx],online_means[idx,pos])
                return result
            runner.add_requests=add;runner._remove_request=remove
            runner.sampler=MotivationReplaySampler(native,buffers)
            runner.steer_vector_state.observe_sample=observe
            params=[SamplingParams(temperature=.7,top_p=.95,seed=42,max_tokens=len(r['token_ids']),ignore_eos=True,skip_special_tokens=False) for r in rows]
            t0=time.monotonic()
            ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=params,steering=steer if group=='R' else None,use_tqdm=False)
            pending.update(zip(ids,rows));ops=llm.llm_engine.output_processor.request_states
            mapping={ops[r].external_req_id:r for r in ids};answers={}
            with (args.output/(group+'_partial.jsonl')).open('x') as f:
                while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    for output in llm.llm_engine.step():
                        assert output.finished;rid=mapping[output.request_id];tokens=list(output.outputs[0].token_ids)
                        assert tokens==pending[rid]['token_ids'];answers[rid]=tokens
                        # Preserve every finished trajectory even if a later one fails.
                        f.write(json.dumps(dict(request_id=rid,token_ids=tokens,status='tokens_complete'))+'\n');f.flush()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
                    rec=dict(pending[rid],**completed[rid]);f.write(json.dumps(rec)+'\n');f.flush()
            torch.cuda.synchronize();elapsed=time.monotonic()-t0
            assert len(answers)==len(rows) and not active
            runner.sampler=native;runner.add_requests=original_add;runner._remove_request=original_remove
            runner.steer_vector_state.observe_sample=original_observe
            summary.append(dict(group=group,rows=len(rows),tokens=sum(map(len,answers.values())),replay_seconds=elapsed,native_raw_checks=len(raw_checks)))
            save(args.output/'progress.json',summary)
        save(args.output/'complete.json',dict(passed=True,status='token and native step checks passed; full also requires separate independent forward receipt',
            groups=summary,total_seconds=time.monotonic()-started,release_sha256=sha(home/'release.json')))
    except BaseException:
        save(args.output/'failure.json',dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
