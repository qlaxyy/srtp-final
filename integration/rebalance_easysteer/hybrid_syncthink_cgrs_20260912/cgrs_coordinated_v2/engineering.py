"""Fixed 8x5 native async engineering gate; no accuracy/effect grading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p, canonical=False):
    b=Path(p).read_bytes()
    return hashlib.sha256(b.replace(b'\r\n',b'\n') if canonical else b).hexdigest()


def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--gpu-authorized',action='store_true')
    args=parser.parse_args()
    if not args.gpu_authorized:
        raise ValueError('Record new bounded GPU authorization before running')
    plan=read(args.plan)
    assert plan['arms']==['R','Roff','Rshadow','RC','Clex'] and len(plan['rows'])==8
    assert plan['run_id']=='cgrs_coordinated_v2_engineering8_run1_20260913'
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name,True)==digest,name
    a=plan['assets']
    for name,meta in a['model_files'].items():
        assert sha(Path(a['model_path'])/name)==meta['sha256']
    for key in ('vector','fit'):
        assert sha(a[key]['path'])==a[key]['sha256']
    for row in plan['rows']:
        assert row['split']=='train'
        text=''.join(unicodedata.normalize('NFKC',row['problem']).split())
        assert hashlib.sha256(text.encode()).hexdigest()==row['problem_sha256']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not apps.strip(),'GPU must be exclusive'
    out=args.output_root/plan['run_id'];out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',plan)
    started=time.monotonic();results={}
    try:
        check=subprocess.run([sys.executable,str(HERE/'test_native.py')],capture_output=True,text=True,timeout=60)
        save(out/'native_cpu_check.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
        assert check.returncode==0,'Native CPU tensor checks failed'
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        os.environ['PYTHONNOUSERSITE']='1'
        sys.path[:0]=[str(ROOT/'sources/EasySteer/vllm-steer'),str(ROOT/'sources/EasySteer'),
                     str(ROOT/'integration/rebalance_easysteer/eval')]
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from adapter import Adapter
        tok=AutoTokenizer.from_pretrained(a['model_path'],local_files_only=True)
        boundaries=sorted(i for p,i in tok.get_vocab().items() if 'ĊĊ' in p)
        steer=SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',
            data=from_pt_direction(a['vector']['path'],layers=[20]),algorithm='rebalance',
            scale=1.,layers=[20],normalize=False,apply=ApplySpec(generation_tokens=boundaries),
            params=dict(read(a['fit']['path'])['parameters'],boundary_token_ids=boundaries,
                        think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=a['model_path'],dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=32768,max_num_seqs=256,max_num_batched_tokens=32768,
            gpu_memory_utilization=.9,enable_steer_vector=True,steer_algorithms=['rebalance'],
            steer_graph_mode='in_graph',enforce_eager=False,enable_chunked_prefill=False,
            enable_prefix_caching=False,async_scheduling=True,seed=42)
        core=llm.llm_engine.engine_core.engine_core
        runner=core.model_executor.driver_worker.worker.model_runner
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in plan['rows']]
        assert max(map(len,prompts))+256<32768
        startup=time.monotonic()-started
        for arm in plan['arms']:
            folder=out/arm;folder.mkdir()
            mode={'R':'off','Roff':'off','Rshadow':'shadow','RC':'negative','Clex':'always'}[arm]
            adapter=None if arm=='R' else Adapter(llm,tok,mode=mode,gate_on=True)
            state=runner.steer_vector_state;remove=state.remove_request;histories={}
            def capture(rid,manager):
                if rid in state._dynamic_indices:
                    i=state._dynamic_indices[rid]
                    histories[rid]=hashlib.sha256(state._history[i,:state._history_lengths[rid]]
                                                  .cpu().numpy().tobytes()).hexdigest()
                return remove(rid,manager)
            state.remove_request=capture
            generated={};began=time.monotonic()
            try:
                ids=llm.enqueue([dict(prompt_token_ids=p) for p in prompts],
                    sampling_params=SamplingParams(temperature=.7,top_p=.95,seed=42,
                        max_tokens=256,skip_special_tokens=False),
                    steering=None if arm=='Clex' else steer,use_tqdm=False)
                ops=llm.llm_engine.output_processor.request_states
                mapping={ops[rid].external_req_id:rid for rid in ids}
                save(folder/'request_mapping.json',{
                    rid:dict(train_index=row['train_index'],problem_sha256=row['problem_sha256'])
                    for rid,row in zip(ids,plan['rows'])})
                with (folder/'partial.jsonl').open('x',encoding='utf8') as stream:
                    while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                        if time.monotonic()-began>180:
                            raise TimeoutError('180 second arm ceiling; retain partials')
                        for output in llm.llm_engine.step():
                            assert output.finished,'Must retain native FINAL_ONLY publishing'
                            rid=mapping[output.request_id];answer=output.outputs[0]
                            assert rid not in generated
                            record=dict(token_ids=list(answer.token_ids),finish_reason=answer.finish_reason)
                            generated[rid]=record
                            stream.write(json.dumps(dict(request_id=rid,**record))+'\n');stream.flush()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:
                        runner._remove_request(rid)
                torch.cuda.synchronize()
                records=[dict(row,**generated[rid],R_history_sha256=histories.get(rid))
                         for row,rid in zip(plan['rows'],ids)]
                results[arm]=dict(records=records,generation_seconds=time.monotonic()-began,
                    events=adapter.completed if adapter and adapter.enabled else {})
                save(folder/'result.json',results[arm])
            finally:
                state.remove_request=remove
                if adapter:adapter.close()
            if arm in ('Roff','Rshadow'):
                key=lambda r:[(x['token_ids'],x['R_history_sha256']) for x in r['records']]
                assert key(results[arm])==key(results['R']),arm+' exact equivalence failed'
            if arm=='Rshadow':
                assert sum(x['eligible'] for x in results[arm]['events'].values())>0,'No gate coverage'
            if arm in ('RC','Clex'):
                assert sum(x['changed'] for x in results[arm]['events'].values())>0,'No intervention coverage'
        save(out/'engineering_gate.json',dict(passed=True,arms=plan['arms'],
            source_sha256=plan['source_sha256'],assets=a,plan_sha256=sha(args.plan),
            startup_seconds=startup,wall_seconds=time.monotonic()-started,
            note='8 request async integration only, no effectiveness or full-load throughput claim'))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),completed_arms=list(results),
                                    wall_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
