"""Independent unmodified vLLM prompt scoring versus decode replay, U only.

Diagnostic threshold remains the previous 0.03 absolute probability bound.
Passing cannot turn the failed cross-engine check into historical equivalence.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();home=Path(__file__).parent
    release=json.loads((home/'release.json').read_text());model=release['assets']['model_path']
    for n,h in release['runtime_source_sha256'].items():
        assert hashlib.sha256((a.runtime_root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',PYTHONNOUSERSITE='1')
    sys.path[1:1]=[str(a.runtime_root/'sources/EasySteer/vllm-steer')]
    from vllm import LLM,SamplingParams
    start=time.monotonic()
    llm=LLM(model=model,dtype='bfloat16',max_model_len=32768,max_num_seqs=256,
        max_num_batched_tokens=32768,gpu_memory_utilization=.9,enable_prefix_caching=False,
        enable_chunked_prefill=False,async_scheduling=True,seed=42,
        enable_steer_vector=True,steer_algorithms=['rebalance'],steer_graph_mode='in_graph')
    rows=[json.loads(s) for s in a.replay.read_text().splitlines() if 'logmax' in json.loads(s)]
    n=32
    prompts=[dict(prompt_token_ids=r['prompt_token_ids']+r['token_ids'][:n]) for r in rows]
    outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)
    records=[]
    for r,out in zip(rows,outputs):
        k=len(r['prompt_token_ids']);lp=[max(x.logprob for x in out.prompt_logprobs[j].values()) for j in range(k,k+n)]
        ref=np.exp(r['logmax'][:n]);prob=np.exp(lp);diff=np.abs(prob-ref)
        records.append(dict(train_index=r['train_index'],logmax=lp,max_error=float(diff.max()),mae=float(diff.mean()),
            shifted_mae=float(np.abs(prob[:-1]-ref[1:]).mean()),passed=bool(diff.max()<=.03)))
    a.output.write_text(json.dumps(dict(passed=all(r['passed'] and r['mae']<r['shifted_mae'] for r in records),
        records=records,seconds=time.monotonic()-start,tolerance=.03,replay_sha256=hashlib.sha256(a.replay.read_bytes()).hexdigest(),
        description='Native prompt full-vocabulary top1 probabilities vs cached decode; no sampler hooks'),indent=2))
if __name__=='__main__':main()
