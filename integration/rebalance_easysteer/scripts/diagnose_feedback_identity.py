"""Isolate compiled vs eager feedback-off identity on the fixed8 short inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
from mechanism_candidates import ROOT,BASE,read,save,sha,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--arm',choices=['original_dynamic','feedback_disabled'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bundle=a.bundle.resolve();out=a.output.resolve();plan=read(bundle/'plan.json');item=plan['engineering']
    require(not out.exists(),'Diagnostic exists');require(item['count']==8 and item['max_tokens']==256,'Changed diagnostic scope')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    for name,digest in plan['source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Prototype source changed')
    require(sha(bundle/item['dataset_file'])==item['dataset_sha256'],'Engineering data changed')
    rows=[json.loads(s) for s in (bundle/item['dataset_file']).read_text(encoding='utf-8').splitlines()]
    import sys
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    sys.path.insert(0,str(ROOT/BASE/'eval'))
    import numpy as np
    import torch
    from vllm import LLM,SamplingParams
    from vllm.steer_vectors import ApplySpec,SteeringSpec,VectorSpec,SelectSpec
    from vllm.steer_vectors.payloads import FeedbackDirection
    from easysteer.vectors import from_pt_direction
    from transformers import AutoTokenizer
    from rebalance_static_eval import build_prompt
    from vllm.capture import deserialize_captured
    from easysteer.hidden_states.capture_result import CaptureResult
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    boundary=sorted(i for t,i in tokenizer.get_vocab().items() if 'ĊĊ' in t)
    params=dict(plan['dynamic_parameters'],boundary_token_ids=boundary,
        think_start_token_id=tokenizer.encode('<think>',add_special_tokens=False)[0],think_end_token_id=tokenizer.encode('</think>',add_special_tokens=False)[0])
    asset=plan['arms']['original_dynamic'];vector=bundle/asset['directory']/'auto_vector.pt'
    require(sha(vector)==asset['vector_sha256'],'Vector changed');payload=from_pt_direction(str(vector),layers=[20]);algorithm='rebalance'
    if a.arm=='feedback_disabled':
        feedback_path=bundle/plan['arms']['latent_feedback_clip']['feedback_config'];f=read(feedback_path)
        require(sha(feedback_path)==plan['arms']['latent_feedback_clip']['feedback_config_sha256'],'Metadata changed')
        require(sha(feedback_path.parent/f['readout_file'])==f['readout_sha256'],'Readout changed')
        payload=FeedbackDirection(payload.layers[20],np.load(feedback_path.parent/f['readout_file']),f['negative_centroid_score'],layer=20,enabled=False);algorithm='rebalance_feedback'
    spec=SteeringSpec(vectors=[VectorSpec(data=payload,algorithm=algorithm,layers=[20],scale=1.,normalize=False,apply=ApplySpec(generation_tokens=boundary),params=params)])
    out.mkdir(parents=True)
    ledger=dict(status='loading_model',arm=a.arm,started_unix=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sha(Path(__file__),source=True),plan_sha256=sha(bundle/'plan.json'),purpose='Existing capture API forces raw eager forwards; determine whether disabled mismatch is specific to compiled execution. No efficacy grading.')
    save(out/'ledger.json',ledger)
    try:
        started=time.perf_counter();runtime=plan['runtime']
        llm=LLM(model=plan['model'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,
            gpu_memory_utilization=.9,enable_steer_vector=True,steer_algorithms=[algorithm],steer_graph_mode='in_graph',enforce_eager=False,
            enable_chunked_prefill=False,enable_prefix_caching=False,async_scheduling=True,seed=42)
        ledger['startup_seconds']=time.perf_counter()-started
        rpc=llm.llm_engine.collective_rpc
        rpc('start_capture',args=('hidden_states',),kwargs=dict(layers=[20],dtype='float32',select=SelectSpec(generation='all').to_wire()))
        started=time.perf_counter()
        outputs=llm.generate([build_prompt(tokenizer,r['problem']) for r in rows],sampling_params=SamplingParams(max_tokens=256,temperature=.7,top_p=.95,seed=42,skip_special_tokens=True),steering=spec,use_tqdm=False)
        elapsed=time.perf_counter()-started
        raw=rpc('fetch_captured',args=('hidden_states',),kwargs=dict(clear=True))[0]
        rpc('stop_capture',args=('hidden_states',))
        tensors,metadata=deserialize_captured(raw);captured=CaptureResult(tensors,metadata,outputs)
        records=[]
        for i,output in enumerate(outputs):
            ids=list(output.outputs[0].token_ids)
            require(len(ids)<=256,'Cap exceeded')
            data=captured.sample(i)[20].float().numpy();np.save(out/f'layer20_{i}.npy',data)
            records.append(dict(index=i,train_index=rows[i]['train_index'],token_ids=ids,capture_positions=captured.sample_positions(i),capture_token_ids=captured.sample_token_ids(i),capture_rows=len(data)))
        save(out/'records.json',records)
        ledger.update(status='completed_diagnostic_no_efficacy_claim',completed_unix=time.time(),generation_seconds_including_capture=elapsed,
            short_outputs=len(outputs),total_tokens=sum(len(r['token_ids']) for r in records),capture_rows=sum(r['capture_rows'] for r in records),
            files_sha256={f.name:sha(f) for f in out.iterdir() if f.name!='ledger.json'})
    except BaseException as error:
        ledger.update(status='incomplete',error=repr(error),stopped_unix=time.time());raise
    finally:save(out/'ledger.json',ledger)
    print(json.dumps(ledger))


if __name__=='__main__':main()
