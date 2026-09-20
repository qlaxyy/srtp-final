"""Opt-in diagnostic harness; reuse the frozen runner, never edit its controller.

CPU --check only validates the bounded harness transformation. GPU --execute
requires the existing isolated runtime and a separately authorized batch.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import textwrap
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NS=HERE.parents[1]
RUN_ID='prebias_gsm64_trace512_run1_20260913'


def patched_child(source):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_child')
    tail=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='stage' for t in n.targets))
    node.body=node.body[:tail]+ast.parse(textwrap.dedent("""
    for diagnostic_arm, diagnostic_mode in [('R','off'),('shadow_R','shadow'),('RS','mix05')]:
        DIAG.begin(diagnostic_arm)
        group('gsm8k_mix05',diagnostic_arm,True,diagnostic_mode)
        DIAG.end(diagnostic_arm)
    llm.llm_engine.engine_core.engine_core.shutdown()
    """)).body
    install=ast.parse('DIAG.install(runner)').body[0]
    index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='history' for t in n.targets))
    node.body.insert(index,install)
    group=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='group')
    caps=[n for n in group.body if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='cap' for t in n.targets)]
    assert len(caps)==1 and isinstance(caps[0].value,ast.IfExp)
    caps[0].value=ast.Constant(512)
    # Retain all ReBalance scale histories for this diagnostic batch only.
    class History(ast.NodeTransformer):
        def visit_IfExp(self,n):
            if isinstance(n.body,ast.Call) and isinstance(n.body.func,ast.Attribute) and n.body.func.attr=='tolist':
                return n.body
            return self.generic_visit(n)
    node=History().visit(node)
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))


class Trace:
    def __init__(self,output,prompts,indices,reference):
        self.output=output;self.prompts=prompts;self.indices=indices
        self.reference=reference
        self.group=None;self.batch=None;self.step=0

    def begin(self,group):
        if hasattr(self,'stream'):self.stream.close()
        self.group=group;self.step=0
        self.stream=(self.output/(group+'_trace.jsonl')).open('x',encoding='utf-8')

    def end(self,group):
        self.stream.flush()
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))['records']
        current=read(self.output/'gsm8k_mix05'/group/'result.json')
        if group=='R':
            previous=read(self.reference/'gsm8k_mix05/R/result.json')
            assert all(x['train_index']==y['train_index'] and x['token_ids']==y['token_ids'][:512]
                       for x,y in zip(current,previous)),'diagnostic R prefix differs from original run'
        if group=='shadow_R':
            base=read(self.output/'gsm8k_mix05/R/result.json')
            assert all(x['token_ids']==y['token_ids'] and x['R_history']['sha256']==y['R_history']['sha256']
                       for x,y in zip(current,base)),'shadow changes original behavior'

    def install(self,runner):
        import torch
        import numpy as np
        from vllm.v1.worker.gpu.sample.hybrid_termination import HybridTerminationState
        self.runner=runner
        original_sample=runner.sample
        original_logits=runner.model.compute_logits
        original_read=HybridTerminationState.read_apply
        original_filter=HybridTerminationState.apply_after_filter

        def guard_read(state,logits,batch):
            for p,rid in enumerate(batch.req_ids):
                if rid in state.requests:
                    assert state.requests[rid][0]==int(batch.idx_mapping_np[p]),'hybrid slot mismatch'
            before=logits.clone()
            ticket=original_read(state,logits,batch)
            torch._assert_async(torch.eq(before,logits).all(),'read-only trigger changed raw logits')
            return ticket

        def guard_filter(state,logits):
            before=logits.clone()
            ticket=state.pending_soft
            result=original_filter(state,logits)
            if ticket is not None:
                pos,idx,trigger=ticket
                changed=(result!=before)
                changed[:,151649]=False
                torch._assert_async(~changed.any(),'non-end logit changed')
                end_changed=result[pos,151649]!=before[pos,151649]
                torch._assert_async((~end_changed|trigger).all(),'untriggered end logit changed')
                all_rows=torch.zeros(logits.shape[0],dtype=torch.bool,device=logits.device)
                all_rows[pos]=True
                torch._assert_async(((result[:,151649]==before[:,151649])|all_rows).all(),'unowned row changed')
            return result

        def capture_logits(hidden,*args,**kwargs):
            logits=original_logits(hidden,*args,**kwargs)
            batch=self.batch
            if batch is None:return logits
            selected=[]
            for row,rid in enumerate(batch.req_ids):
                idx=int(batch.idx_mapping_np[row])
                # Identity comes from the request's frozen prompt, registered below.
                problem=self.request_ids.get(rid)
                generated=int(batch.num_computed_tokens_np[row]+batch.num_scheduled_tokens[row]-runner.req_states.prompt_len.np[idx])
                if problem==5424 and 408<=generated<=418:
                    generated=int(batch.seq_lens[row].cpu())-int(runner.req_states.prompt_len.np[idx])
                    if 409<=generated<=417:selected.append((row,idx,generated,rid))
            for row,idx,generated,rid in selected:
                name=f'{self.group}_5424_t{generated}'
                with (self.output/(name+'_logits.npy')).open('xb') as f:
                    np.save(f,logits[row].detach().float().cpu().numpy(),allow_pickle=False)
                with (self.output/(name+'_hidden.npy')).open('xb') as f:
                    np.save(f,hidden[row].detach().float().cpu().numpy(),allow_pickle=False)
                state=runner.steer_vector_state
                h=getattr(runner,'hybrid_termination',None)
                fields={k:float(getattr(state,k)[idx].cpu()) for k in ('_coefs','_step_prob_sum','_step_tok_count','_prev_step_mean','_in_think')}
                data=dict(group=self.group,train_index=5424,generated_position=generated,slot=idx,
                    prompt_length=int(runner.req_states.prompt_len.np[idx]),row=row,step=self.step,
                    num_reqs=batch.num_reqs,padded_reqs=batch.num_reqs_after_padding,
                    padded_tokens=batch.num_tokens_after_padding,R_state=fields,
                    sampling_seed=int(runner.sampler.sampling_states.seeds.gpu[idx].cpu()),
                    input_position=int(batch.positions[batch.logits_indices[row]].cpu()),
                    first_bias=int(h.first_bias[idx]) if h else -1)
                with (self.output/(name+'.json')).open('x',encoding='utf-8') as f:
                    f.write(json.dumps(data,indent=2)+'\n')
            return logits

        def traced_sample(hidden,batch,grammar):
            self.batch=batch
            expected_slots=torch.as_tensor(batch.idx_mapping_np.copy(),device=batch.idx_mapping.device)
            torch._assert_async((batch.idx_mapping==expected_slots).all(),'GPU/CPU slot mapping differs')
            for rid in batch.req_ids:
                assert runner.steer_vector_state._dynamic_indices[rid]==runner.req_states.req_id_to_index[rid],'R slot mismatch'
                if rid not in self.request_ids:
                    slot=runner.req_states.req_id_to_index[rid]
                    n=int(runner.req_states.prompt_len.np[slot])
                    ids=runner.req_states.all_token_ids.gpu[slot,:n].cpu().tolist()
                    matches=[i for i,p in zip(self.indices,self.prompts) if p==ids]
                    assert len(matches)==1,'request identity mismatch'
                    self.request_ids[rid]=matches[0]
            row=dict(group=self.group,step=self.step,req_ids=batch.req_ids,
                train_indices=[self.request_ids[rid] for rid in batch.req_ids],
                slots=batch.idx_mapping_np.tolist(),num_reqs=batch.num_reqs,
                padded_reqs=batch.num_reqs_after_padding,padded_tokens=batch.num_tokens_after_padding,
                computed=batch.num_computed_tokens_np.tolist(),scheduled=batch.num_scheduled_tokens.tolist())
            self.stream.write(json.dumps(row)+'\n');self.stream.flush()
            result=original_sample(hidden,batch,grammar)
            self.step+=1;self.batch=None
            return result

        self.request_ids={}
        HybridTerminationState.read_apply=guard_read
        HybridTerminationState.apply_after_filter=guard_filter
        runner.model.compute_logits=capture_logits
        runner.sample=traced_sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resolved',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    source=(NS/'run_batch.py').read_text(encoding='utf-8')
    tree=patched_child(source)
    compile(tree,'bounded_diagnostic_runner','exec')
    if a.check:
        print(json.dumps(dict(status='CPU_AST_check_pass',model_initialized=False,groups=3,questions_per_group=64,cap=512,max_new_tokens=98304)))
        return
    assert a.execute and a.resolved and a.output
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
    r=json.loads(a.resolved.read_text(encoding='utf-8'))
    assert r['run_id']=='s64_mix05_math64_gsm64_c256_run1_20260913' and r['mix05']
    expected=r['source_sha256']['integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/run_batch.py']
    assert hashlib.sha256((NS/'run_batch.py').read_bytes()).hexdigest()==expected
    a.output.mkdir(parents=True,exist_ok=False)
    reference=Path(r['output'])
    r['run_id']=RUN_ID;r['output']=str(a.output);r['expansion']['arm_budget_seconds']=180
    resolved=a.output/'diagnostic_resolved.json'
    resolved.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    spec=importlib.util.spec_from_file_location('frozen_batch',NS/'run_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,meta in r['assets']['model_files'].items():
        assert module.sha(Path(r['assets']['model_path'])/name)==meta['sha256'],name
    for name in ('vector','fit'):
        assert module.sha(r['assets'][name]['path'])==r['assets'][name]['sha256'],name
    (a.output/'instrumentation_identity.json').write_text(json.dumps(dict(
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformed_runner_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),
        baseline_runtime_commit=r['deployment']['commit'],
        assets_sha256_verified=True,mode='diagnostic_only',formal_effect_test=False),indent=2)+'\n',encoding='utf-8')
    trace=Trace(a.output,r['prompts']['gsm8k_mix05'],[x['train_index'] for x in r['soft2_rows']['gsm8k_mix05']],reference)
    module.__dict__['DIAG']=trace
    exec(compile(tree,'bounded_diagnostic_runner','exec'),module.__dict__)
    started=time.monotonic();status='failed'
    try:
        module.run_child(SimpleNamespace(resolved_plan=resolved,child='integrated'))
        status='complete'
    finally:
        (a.output/'diagnostic_status.json').write_text(json.dumps(dict(status=status,seconds=time.monotonic()-started,formal_effect_test=False,new_answers_max=192,cap=512,extra_probe_tokens=0,environment_batch_invariant=os.getenv('VLLM_BATCH_INVARIANT','0')),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
