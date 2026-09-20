"""Full fixed benchmark for the two independently recalibrated policies."""
import argparse,hashlib,json,os,signal,sys,time,subprocess
from pathlib import Path
HOME=Path(__file__).resolve().parent
SUPPORT=Path('/root/autodl-tmp/projects/iterative_recalibration_20260919/integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/iterative_recalibration_20260919')
sys.path.insert(0,str(SUPPORT))
from common import read,save,sha,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['R','L27'],required=True)
    p.add_argument('--quantile',choices=['q75','q90'],required=True);p.add_argument('--fit-dir',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering',action='store_true')
    p.add_argument('--engineering-result',type=Path);a=p.parse_args();verify()
    m=read(HOME/'manifest.json')
    for n,h in m['files'].items():assert sha(HOME/n)==h,n
    fit=read(a.fit_dir/'fit.json');col=read(a.fit_dir/'complete.json');assert col['passed'] and not col['engineering']
    assert fit['vector_sha256']==sha(a.fit_dir/'auto_vector.pt')
    identity=dict(case=a.case,quantile=a.quantile,fit_sha256=sha(a.fit_dir/'fit.json'),vector_sha256=fit['vector_sha256'],
        calibration_source_sha256=fit['calibration_source_sha256'],eval_manifest_sha256=sha(HOME/'manifest.json'))
    assert not a.engineering, 'This fixed batch runs complete MATH500 only'
    assert fit['confidence_high_quantile']==dict(q75=.75,q90=.90)[a.quantile]
    expected=read(HOME/'summary.json')['groups'][a.case][a.quantile]
    assert identity['fit_sha256']==expected['fit_sha256'] and identity['vector_sha256']==expected['vector_sha256']
    curve=read(a.fit_dir/'curve_check.json');assert curve['finite'] and curve['max_residual']<1e-8
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'identity.json',identity);began=time.monotonic()
    def timeout(*_):raise TimeoutError('Fixed evaluation budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600 if a.engineering else 1800)
    try:
        source=SUPPORT;assets=read(source/'assets.json')
        for n,h in read(source/'runtime_hashes.json').items():
            assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        for n,v in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==v['sha256']
        sys.path[1:1]=[str(SUPPORT.parent),str(a.runtime_root/'sources/EasySteer/vllm-steer'),
            str(a.runtime_root/'sources/EasySteer'),str(a.runtime_root/'integration/rebalance_easysteer/eval')]
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        import torch,numpy as np,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        from rebalance_static_eval import build_prompt
        from label_alignment_adapter import AlignmentAdapter
        assert Path(vllm.__file__).resolve().is_relative_to(a.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        bounds=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s);layer=fit['decoder_output_layer']
        steer=SteeringSpec(vectors=[VectorSpec(name='native_'+a.case+'_'+a.quantile,data=from_pt_direction(str(a.fit_dir/'auto_vector.pt'),layers=[layer]),
            algorithm='rebalance',scale=1.,layers=[layer],normalize=False,apply=ApplySpec(generation_tokens=bounds),
            params=dict(fit['parameters'],boundary_token_ids=bounds,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=256,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
            enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph',
            enforce_eager=False,enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42)
        core=llm.llm_engine.engine_core.engine_core;runner=core.model_executor.driver_worker.worker.model_runner
        rows=read(HOME/'rows.json');assert len(rows)==500
        if a.engineering:rows=rows[:8]
        prompts=[tok.encode(build_prompt(tok,r['problem'])) for r in rows];cap=512 if a.engineering else 16000
        reference=None;history=None;timings=[]
        for repeat in range(2 if a.engineering else 1):
            adapter=None
            if a.case=='L27':
                with np.load(source/'opening.npz') as z:tables={k:z[k] for k in z.files}
                adapter=AlignmentAdapter(llm,tok,tables=tables,large_suppression=True,lexical_control=False,enabled=True)
            generated={};torch.cuda.synchronize();started=time.monotonic()
            ids=llm.enqueue([dict(prompt_token_ids=x) for x in prompts],sampling_params=SamplingParams(
                temperature=.7,top_p=.95,seed=42,max_tokens=cap,skip_special_tokens=False),steering=steer,use_tqdm=False)
            ops=llm.llm_engine.output_processor.request_states
            mapping={ops[r].external_req_id:r for r in ids};rowmap=dict(zip(ids,rows))
            with (a.output/f'partial_{repeat}.jsonl').open('x',encoding='utf-8') as f:
                while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    for output in llm.llm_engine.step():
                        assert output.finished
                        rid=mapping[output.request_id];ans=output.outputs[0];ts=list(ans.token_ids)
                        assert rid not in generated and len(ts)<=cap
                        rec=dict(rowmap[rid],prompt_token_ids=prompts[rows.index(rowmap[rid])],token_ids=ts,tokens=len(ts),
                            thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),
                            text=tok.decode(ts,skip_special_tokens=True),finish_reason=ans.finish_reason)
                        generated[rid]=rec;f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                llm.llm_engine.step()
                for rid in ids:
                    if rid in runner.req_states.req_id_to_index:runner._remove_request(rid)
            torch.cuda.synchronize();timings.append(time.monotonic()-started)
            records=[generated[i] for i in ids];tokens=[r['token_ids'] for r in records]
            events=adapter.completed if adapter is not None else None
            if reference is not None:
                assert tokens==reference,'Repeat tokens differ'
                if adapter is not None:assert [events[i]['R_history_sha256'] for i in ids]==history
            reference=tokens
            if adapter is not None:
                history=[events[i]['R_history_sha256'] for i in ids];adapter.close()
        save(a.output/'result.json',dict(status='complete',records=records,events=events,generation_seconds=timings,
            extra_model_forward_count=0,probe_count=0,timing_note='Includes output collection and checkpoint I/O; no separate pure-kernel measurement'))
        save(a.output/'complete.json',dict(passed=True,engineering=a.engineering,identity=identity,rows=len(rows),
            wall_seconds=time.monotonic()-began,generation_seconds=timings))
    except BaseException as e:
        save(a.output/'failure.json',dict(error=repr(e),wall_seconds=time.monotonic()-began));raise

if __name__=='__main__':main()
