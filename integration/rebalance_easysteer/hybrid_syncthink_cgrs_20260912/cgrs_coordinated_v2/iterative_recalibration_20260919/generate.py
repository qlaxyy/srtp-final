"""One R-only calibration batch or bounded engine check; no shared edits."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path
from common import HOME,HERE,read,save,sha,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering',action='store_true')
    p.add_argument('--engineering-result',type=Path)
    a=p.parse_args();verify();cfg=read(HOME/'plan.json');assets=read(HOME/'assets.json')
    if not a.engineering:
        gate=read(a.engineering_result/'complete.json')
        assert gate['passed'] and gate['engineering'] and gate['manifest_sha256']==sha(HOME/'manifest.json')
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    def timeout(*_):raise TimeoutError('Generation ceiling; keep partial answers')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600 if a.engineering else 2100)
    try:
        import hashlib
        for n,h in read(HOME/'runtime_hashes.json').items():
            assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
        for n,m in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==m['sha256']
        for k in ('fit','vector'):assert sha(assets[k]['path'])==assets[k]['sha256']
        sys.path[1:1]=[str(HERE),str(a.runtime_root/'sources/EasySteer/vllm-steer'),str(a.runtime_root/'sources/EasySteer')]
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.steer_vectors import SteeringSpec,VectorSpec,ApplySpec
        from easysteer.vectors import from_pt_direction
        assert Path(vllm.__file__).resolve().is_relative_to(a.runtime_root/'sources/EasySteer/vllm-steer')
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        bounds=sorted(i for s,i in tok.get_vocab().items() if 'ĊĊ' in s)
        layer=assets['decoder_output_layer'];fit=read(assets['fit']['path'])
        assert fit['decoder_output_layer']==layer
        steer=SteeringSpec(vectors=[VectorSpec(name='iteration_R_parent',
            data=from_pt_direction(assets['vector']['path'],layers=[layer]),algorithm='rebalance',scale=1.,
            layers=[layer],normalize=False,apply=ApplySpec(generation_tokens=bounds),
            params=dict(fit['parameters'],boundary_token_ids=bounds,think_start_token_id=151648,think_end_token_id=151649))])
        llm=LLM(model=assets['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=cfg['max_num_seqs'],max_num_batched_tokens=cfg['max_num_batched_tokens'],
            gpu_memory_utilization=cfg['gpu_memory_utilization'],enable_steer_vector=True,
            steer_algorithms=['rebalance'],steer_graph_mode='in_graph',enforce_eager=False,
            enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42)
        core=llm.llm_engine.engine_core.engine_core
        rows=read(HOME/'rows.json');rows=rows[:8] if a.engineering else rows
        cap=512 if a.engineering else 16000
        reference=None;seconds=[]
        for run in range(2 if a.engineering else 1):
            generated={};torch.cuda.synchronize();start=time.monotonic()
            ids=llm.enqueue([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows],
                sampling_params=SamplingParams(temperature=0,top_p=1,seed=42,max_tokens=cap,skip_special_tokens=False),
                steering=steer,use_tqdm=False)
            ops=llm.llm_engine.output_processor.request_states
            mapping={ops[r].external_req_id:r for r in ids};rowmap=dict(zip(ids,rows))
            with (a.output/f'partial_{run}.jsonl').open('x',encoding='utf-8') as f:
                while llm.llm_engine.has_unfinished_requests() or core.batch_queue:
                    for output in llm.llm_engine.step():
                        assert output.finished
                        rid=mapping[output.request_id];ans=output.outputs[0];ts=list(ans.token_ids)
                        assert rid not in generated and len(ts)<=cap
                        rec=dict(rowmap[rid],token_ids=ts,thinking_tokens=ts.index(151649) if 151649 in ts else len(ts),
                            tokens=len(ts),text=tok.decode(ts,skip_special_tokens=True),finish_reason=ans.finish_reason)
                        generated[rid]=rec;f.write(__import__('json').dumps(rec,ensure_ascii=False)+'\n');f.flush()
                llm.llm_engine.step()
            torch.cuda.synchronize();seconds.append(time.monotonic()-start)
            records=[generated[i] for i in ids]
            if reference is not None:assert [r['token_ids'] for r in records]==reference
            reference=[r['token_ids'] for r in records]
        save(a.output/'result.json',dict(records=records,generation_seconds=seconds,
            new_answers=len(rows)*(2 if a.engineering else 1),extra_model_forwards=0,method='R only; no lexical adapter'))
        save(a.output/'complete.json',dict(passed=True,engineering=a.engineering,manifest_sha256=sha(HOME/'manifest.json'),
            wall_seconds=time.monotonic()-began,generation_seconds=seconds))
    except BaseException as e:
        save(a.output/'failure.json',dict(error=repr(e),wall_seconds=time.monotonic()-began));raise

if __name__=='__main__':main()
