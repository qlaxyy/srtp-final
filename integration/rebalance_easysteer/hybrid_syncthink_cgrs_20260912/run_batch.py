"""Prepare and execute only the fixed B-line engineering + 64 x 4 screen."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GEN = '/root/autodl-tmp/venvs/easysteer-vllm026/bin/python'
GRADE = '/root/autodl-tmp/venvs/rebalance/bin/python'
sys.path.insert(0, str(ROOT/'integration/rebalance_easysteer/eval'))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['PYTHONNOUSERSITE'] = '1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def dataset(role):
    return [json.loads(s) for s in (HERE/'prepared_run1'/f'{role}.jsonl').read_text(encoding='utf-8').splitlines()]


def prepare(a):
    from transformers import AutoTokenizer
    from rebalance_static_eval import build_prompt
    plan = read(HERE/'experiment_plan.json')
    freeze = read(HERE/'prepared_run1/data_freeze.json')
    assert freeze['registry_sha256'] == sha(HERE/'cpu_run1/data_registry.json')
    assert sha(HERE/'prepared_run1/data_audit.json') == freeze['audit_sha256']
    assert read(HERE/'prepared_run1/data_audit.json')['global_local_snapshot_clear']
    model = Path(plan['assets']['model_path'])
    for name, meta in plan['assets']['model_files'].items():
        assert sha(model/name) == meta['sha256'], name
    for key in ('vector','fit'):
        assert sha(plan['assets'][key]['path']) == plan['assets'][key]['sha256'], key
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = {}
    for role,n in [('engineering',8),('screening',64)]:
        assert sha(HERE/'prepared_run1'/f'{role}.jsonl') == freeze['role_files'][role]
        rows = dataset(role)
        assert len(rows) == n
        prompts[role] = [tokenizer.encode(build_prompt(tokenizer,r['problem'])) for r in rows]
        assert max(map(len,prompts[role])) + 16000 < 32768
    files = {}
    for folder in ('sources/EasySteer/vllm-steer/vllm','sources/EasySteer/easysteer',
                   'sources/ReBalance/utils','integration/rebalance_easysteer/eval',
                   'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.jsonl'):
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    # Runtime files arrive from one immutable local archive with receipt.
    import torch, vllm
    assert sha(ROOT/'baseline_model_runner.py') == '88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5'
    files['baseline_model_runner.py'] = sha(ROOT/'baseline_model_runner.py')
    resolved = dict(run_id=a.run_id, status='ready_for_authorized_first_batch',
                    code_commit=read(ROOT/'deployment_identity.json')['commit'],
                    deployment=read(ROOT/'deployment_identity.json'),
                    data_frozen=True, remote_inventory=read(ROOT/'remote_inventory.json'),
                    data_registry_sha256=freeze['registry_sha256'], source_sha256=files,
                    plan_sha256=sha(HERE/'experiment_plan.json'), prompts=prompts,
                    model_files=plan['assets']['model_files'], assets=plan['assets'],
                    environment=dict(python=sys.version,torch=torch.__version__,vllm=vllm.__version__,
                                     vllm_path=vllm.__file__),
                    output='/root/autodl-tmp/results/easysteer/'+plan['namespace']+'/'+a.run_id,
                    scope='8 engineering x 5, then only 64 x 4 if engineering passes',
                    authorization='User: 请继续，进行测试; first batch only',
                    preemption_policy='Reject any actual preemption; restore not implemented in first batch',
                    prepared_unix=time.time())
    assert resolved['remote_inventory']['new_unreported_result_files'] == []
    save(a.resolved_plan,resolved)
    print('PREPARED',sha(a.resolved_plan),flush=True)


def validate(resolved):
    assert resolved['data_frozen'] and resolved['run_id']
    assert sha(HERE/'experiment_plan.json') == resolved['plan_sha256']
    for name,h in resolved['source_sha256'].items():
        assert sha(ROOT/name) == h, name


def run_child(a):
    r = read(a.resolved_plan)
    validate(r)
    output = Path(r['output'])
    if a.child == 'pre':
        name = 'vllm.v1.worker.gpu.model_runner'
        assert name not in sys.modules
        spec = importlib.util.spec_from_file_location(name,ROOT/'baseline_model_runner.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from easysteer.vectors import from_pt_direction
    from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
    from rebalance_static_eval import build_prompt
    tokenizer = AutoTokenizer.from_pretrained(r['assets']['model_path'],local_files_only=True)
    boundaries = sorted(i for t,i in tokenizer.get_vocab().items() if 'ĊĊ' in t)
    fit = read(r['assets']['fit']['path'])
    steering = SteeringSpec(vectors=[VectorSpec(name='rebalance_dynamic',
        data=from_pt_direction(r['assets']['vector']['path'],layers=[20]),
        algorithm='rebalance',scale=1.,layers=[20],normalize=False,
        apply=ApplySpec(generation_tokens=boundaries),
        params=dict(fit['parameters'],boundary_token_ids=boundaries,
                    think_start_token_id=151648,think_end_token_id=151649))])
    started = time.monotonic()
    llm = LLM(model=r['assets']['model_path'],dtype='bfloat16',tensor_parallel_size=1,
              max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,
              gpu_memory_utilization=.9,enable_steer_vector=True,steer_algorithms=['rebalance'],
              enforce_eager=False,steer_graph_mode='in_graph',enable_chunked_prefill=False,
              enable_prefix_caching=False,async_scheduling=False,seed=42)
    startup = time.monotonic()-started
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    if not runner.steer_vector_state.supports_kv_replay:
        raise RuntimeError('Wrong model runner')
    def preempt(*args,**kwargs):
        raise RuntimeError('Batch stops before any unvalidated preemption')
    core.scheduler._preempt_request = preempt
    history = {}
    original_remove = runner.steer_vector_state.remove_request
    state = runner.steer_vector_state
    def capture(rid,manager):
        if rid in state._dynamic_indices:
            idx = state._dynamic_indices[rid]
            length = state._history_lengths[rid]
            values = state._history[idx,:length].cpu().numpy()
            history[rid] = dict(sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                                length=length, values=values.tolist() if a.child=='pre' or stage=='engineering' else None)
        return original_remove(rid,manager)
    state.remove_request = capture

    def group(stage,name,use_r,mode):
        history.clear()
        if hasattr(runner,'hybrid_termination'):
            runner.hybrid_termination = None
        rows = dataset('engineering' if stage=='engineering' else 'screening')
        cap = 512 if stage=='engineering' else 16000
        folder = output/stage/name
        folder.mkdir(parents=True,exist_ok=False)
        config = dict(mode=mode,entropy_weight=.8,pacing_cap=64,end_token_id=151649)
        sampling = SamplingParams(temperature=.7,top_p=.95,max_tokens=cap,seed=42,
                                  skip_special_tokens=True,
                                  extra_args={'hybrid_syncthink_cgrs_20260912':config} if mode!='absent' else None)
        prompts = [build_prompt(tokenizer,row['problem']) for row in rows]
        actual_prompt_ids = [tokenizer.encode(s) for s in prompts]
        assert actual_prompt_ids == r['prompts']['engineering' if stage=='engineering' else 'screening']
        began = time.monotonic()
        torch.cuda.synchronize()
        request_ids = llm.enqueue(prompts,sampling_params=sampling,steering=steering if use_r else None)
        states = llm.llm_engine.output_processor.request_states
        indices = {states[rid].external_req_id:i for i,rid in enumerate(request_ids)}
        completed = {}
        with (folder/'partial.jsonl').open('x',encoding='utf-8') as f:
            while llm.llm_engine.has_unfinished_requests():
                if time.monotonic()-began > (600 if stage=='screening' else 120):
                    raise TimeoutError('Per-arm wall budget')
                for result in llm.llm_engine.step():
                    if result.finished:
                        i = indices[result.request_id]
                        assert i not in completed
                        generated = result.outputs[0]
                        ids = list(generated.token_ids)
                        count = ids.index(151649) if 151649 in ids else len(ids)
                        record = dict(local_index=i,train_index=rows[i]['train_index'],
                            problem=rows[i]['problem'],gold=rows[i]['answer'],
                            prompt_token_ids=result.prompt_token_ids,token_ids=ids,text=generated.text,
                            tokens=len(ids),thinking_tokens=count,answer_tokens=len(ids)-count-int(151649 in ids),
                            finish_reason=generated.finish_reason,request_id=request_ids[i])
                        completed[i]=record
                        f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
        torch.cuda.synchronize()
        seconds = time.monotonic()-began
        assert len(completed)==len(rows)
        for rid in request_ids:
            if rid in state._dynamic_indices:
                idx=state._dynamic_indices[rid]; length=state._history_lengths[rid]
                values=state._history[idx,:length].cpu().numpy()
                history[rid]=dict(sha256=hashlib.sha256(values.tobytes()).hexdigest(),length=length,
                                  values=values.tolist() if stage=='engineering' else None)
        hybrid=getattr(runner,'hybrid_termination',None)
        receipt=hybrid.finish_all() if hybrid is not None else dict(requests={},statistics_and_mask_gpu_ms=0,extra_model_forwards=0,probe_tokens=0)
        for i,record in completed.items():
            rid=record['request_id']
            record['R_history']=history.get(rid)
            record['hybrid']=receipt['requests'].get(rid)
            if mode in ('shadow','enforce'):
                assert record['hybrid']['accepted_tokens']==record['tokens']
                if mode=='enforce' and record['hybrid']['first_trigger']>=0:
                    assert record['token_ids'][record['hybrid']['first_trigger']]==151649
        save(folder/'result.json',dict(status='complete',stage=stage,arm=name,cap=cap,
             generation_seconds=seconds,startup_seconds=startup,control=receipt,
             records=[completed[i] for i in range(len(rows))],preemptions=0))
        print('COMPLETE',stage,name,len(rows),round(seconds,3),flush=True)

    stage='engineering'
    if a.child=='pre':
        group(stage,'pre_R',True,'absent')
    else:
        for name,use_r,mode in [('off_R',True,'off'),('shadow_R',True,'shadow'),('S',False,'enforce'),('RS',True,'enforce')]:
            group(stage,name,use_r,mode)
        base=read(output/'engineering/pre_R/result.json')['records']
        for name in ('off_R','shadow_R'):
            current=read(output/'engineering'/name/'result.json')['records']
            for x,y in zip(base,current):
                assert x['token_ids']==y['token_ids'], name+' token equality'
                assert x['R_history']['sha256']==y['R_history']['sha256'], name+' R history equality'
        combined=read(output/'engineering/RS/result.json')['records']
        for x,y in zip(base,combined):
            trigger=y['hybrid']['first_trigger']
            n=trigger if trigger>=0 else min(len(x['token_ids']),len(y['token_ids']))
            assert x['token_ids'][:n]==y['token_ids'][:n], 'Divergence before trigger'
        save(output/'engineering_gate.json',dict(status='pass',token_and_R_history_identity=True,
             pretrigger_identity=True,actual_preemption_tested=False))
        stage='screening'
        for name,use_r,mode in [('U',False,'absent'),('R',True,'off'),('S',False,'enforce'),('RS',True,'enforce')]:
            group(stage,name,use_r,mode)
    llm.llm_engine.engine_core.shutdown()


def execute(a):
    r=read(a.resolved_plan); validate(r)
    assert a.run_id==r['run_id'] and a.phase=='screen'
    output=Path(r['output']);output.mkdir(parents=True,exist_ok=False)
    save(output/'resolved_plan.json',r)
    start=time.monotonic()
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'sources/EasySteer/vllm-steer')+':'+str(ROOT/'sources/EasySteer')
    status='failed'
    try:
        for child in ('pre','integrated'):
            command=[GEN,'-u',str(__file__),'--resolved-plan',str(a.resolved_plan),'--child',child]
            with (output/(child+'.log')).open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:
                    while process.poll() is None:
                        elapsed=time.monotonic()-start
                        if elapsed>1800 or (elapsed>300 and not (output/'engineering_gate.json').exists()):
                            raise TimeoutError('Registered whole/engineering budget exhausted')
                        time.sleep(1)
                    if process.returncode:
                        raise RuntimeError(f'{child} exited {process.returncode}')
                finally:
                    if process.poll() is None:
                        import signal
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        subprocess.run([GRADE,str(HERE/'grade_screen.py'),'--output',str(output)],check=True,timeout=max(1,1800-(time.monotonic()-start)))
        status='complete'
    finally:
        save(output/'batch_status.json',dict(status=status,wall_seconds=time.monotonic()-start,
                                            retries=0,scope='first batch only'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resolved-plan',type=Path,required=True)
    p.add_argument('--run-id')
    p.add_argument('--phase',default='screen',choices=['screen'])
    p.add_argument('--child',choices=['pre','integrated'])
    a=p.parse_args()
    if a.prepare_only:prepare(a)
    elif a.child:run_child(a)
    elif a.execute:execute(a)
    else:p.error('Choose --prepare-only or --execute')


if __name__=='__main__':main()
