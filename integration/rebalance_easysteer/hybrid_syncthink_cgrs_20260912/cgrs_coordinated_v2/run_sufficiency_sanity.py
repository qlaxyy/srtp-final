"""Scorer-only controlled-input check. Not an early-exit implementation."""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse_score(text,finish_reason):
    if finish_reason!='stop':return None
    m=re.fullmatch(r'(\d{1,3})',text.strip())
    return int(m[1]) if m and int(m[1])<=100 else None
def save(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());assert len(plan['rows'])==16 and plan['max_tokens']==8
    assert digest(Path(__file__))==plan['runner_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=a.output/plan['run_id'];out.mkdir(parents=True,exist_ok=False);save(out/'plan.json',plan)
    start=time.monotonic()
    def timeout(*_):raise TimeoutError('180 second sanity ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    try:
        for name,meta in plan['model_files'].items():assert digest(Path(plan['model_path'])/name)==meta['sha256']
        source=Path('/root/autodl-tmp/projects/hybrid_wsc_native_long_20260917')
        sys.path[:0]=[str(source/'sources/EasySteer/vllm-steer'),str(source/'sources/EasySteer')]
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['PYTHONNOUSERSITE']='1'
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        import vllm
        assert Path(vllm.__file__).resolve().is_relative_to(source/'sources/EasySteer/vllm-steer')
        tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True)
        prompts=[]
        for row in plan['rows']:
            content=plan['prompt']+'\n\nProblem:\n'+row['problem']+'\n\nProposed reasoning:\n'+row['thought']
            ids=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True)
            assert ids.count(151648)==1 and not tokenizer.decode(ids[ids.index(151648)+1:]).strip(), 'Expected native open-thought prefix, optionally followed by whitespace'
            ids+=tokenizer.encode('</think>\n\nScore: ',add_special_tokens=False)
            assert ids.count(151648)==1 and ids.count(151649)==1 and len(ids)+8<4096
            prompts.append(dict(prompt_token_ids=ids))
        save(out/'inputs.json',dict(prompts=prompts,case_ids=[r['case_id'] for r in plan['rows']],
                                  input_tokens=sum(len(x['prompt_token_ids']) for x in prompts)))
        llm=LLM(model=plan['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,
                max_num_seqs=32,max_num_batched_tokens=8192,gpu_memory_utilization=.80,
                enforce_eager=True,compilation_config=0,enable_prefix_caching=False,
                enable_chunked_prefill=False,async_scheduling=False,seed=42)
        startup=time.monotonic()-start;began=time.monotonic()
        outputs=llm.generate(prompts,sampling_params=SamplingParams(temperature=0,top_p=1,seed=42,max_tokens=8,skip_special_tokens=True),use_tqdm=False)
        generation=time.monotonic()-began;rows=[]
        for row,output in zip(plan['rows'],outputs):
            answer=output.outputs[0];score=parse_score(answer.text,answer.finish_reason)
            rows.append(dict(case_id=row['case_id'],category=row['category'],expected_stop=row['expected_stop'],
                text=answer.text,score=score,token_ids=list(answer.token_ids),finish_reason=answer.finish_reason))
        unsafe=[r['case_id'] for r in rows if not r['expected_stop'] and r['score']==100]
        valid=all(r['score'] is not None for r in rows)
        positives=sum(r['expected_stop'] and r['score']==100 for r in rows)
        save(out/'result.json',dict(rows=rows,all_parse=valid,unsafe_full_scores=unsafe,correct_complete_full_scores=positives,
            passes=valid and not unsafe and positives>=3,total_generated_tokens=sum(len(r['token_ids']) for r in rows),
            startup_seconds=startup,generation_seconds=generation,wall_seconds=time.monotonic()-start,
            plan_sha256=digest(a.plan),runner_sha256=digest(Path(__file__)),
            scope='Synthetic scoring sanity; not compression or natural-prefix evaluation; scores are uncalibrated model outputs.'))
        print(json.dumps(dict(passed=valid and not unsafe and positives>=3,unsafe=unsafe,positives=positives,generation_seconds=generation)))
    except BaseException as exc:
        save(out/'failure.json',dict(error=repr(exc),wall_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
