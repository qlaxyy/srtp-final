"""Instance-only forced-token/frozen-R-state batch geometry diagnostic."""
import json,math
from pathlib import Path

STATE_FIELDS=('_coefs','_step_prob_sum','_step_tok_count','_prev_step_mean',
              '_in_think','_paper_strength','_paper_pending')
PHASES=('reference8','repeat8','single1','refill4plus4')


def logical_geometry(geometry):
    return [dict(num_tokens=g['num_tokens'],padded_tokens=g['padded_tokens'],
        members=[{k:v for k,v in m.items() if k not in ('request_id','slot')} for m in g['members']]) for g in geometry]


def validate_input_slice(prompt,forced,positions,tokens):
    source=prompt+forced
    if len(positions)!=len(tokens) or any(type(p) is not int or p<0 or p>=len(source) for p in positions):
        raise RuntimeError('Invalid input position coverage')
    if any(source[p]!=t for p,t in zip(positions,tokens)):
        raise RuntimeError('Model input differs from fixed prompt/token prefix')


def ready_for_refill(counts,first_ids):
    return bool(first_ids) and all(counts.get(rid,0)>=32 for rid in first_ids)


class FrozenPrefixTrace:
    def __init__(self,llm,folder,rows,reference=None,strict_repeat=False):
        import torch
        self.t=torch;self.folder=Path(folder);self.reference=reference;self.strict_repeat=strict_repeat
        self.core=llm.llm_engine.engine_core.engine_core
        self.runner=self.core.model_executor.driver_worker.worker.model_runner
        self.state=self.runner.steer_vector_state
        self.native=self.runner.sampler;self.original_scales=self.state.token_scales
        self.original_observe=self.state.observe_sample
        self.original_preempt=self.core.scheduler._preempt_request
        self.rows=rows;self.requests={};self.counts={};self.before={};self.records={};self.states={};self.logits={}
        self.geometry=[];self.forward_tickets={};self.state_locks=0;self.native_draws={};self.max_probabilities={}
        self.runner.sampler=self;self.state.token_scales=self.token_scales
        self.state.observe_sample=self.observe_sample
        self.core.scheduler._preempt_request=self.reject_preemption

    @staticmethod
    def reject_preemption(*args,**kwargs):raise RuntimeError('Preemption excluded from fixed-layout diagnostic')

    def __getattr__(self,name):return getattr(self.native,name)

    def register(self,rids,indices):
        if len(rids)!=len(indices):raise RuntimeError('Request registration length mismatch')
        for rid,index in zip(rids,indices):
            if rid in self.requests:raise RuntimeError('Request identity reused')
            self.requests[rid]=index;self.counts[rid]=0

    def key(self,rid):return (self.requests[rid],self.counts[rid])

    def snapshot(self,rid,slot):
        s=self.state;k=self.counts[rid];row=self.rows[self.requests[rid]]
        length=len(row['prompt_token_ids'])+k
        if s._history_lengths[rid]!=length:raise RuntimeError('R history length/accepted-token clock disagreement')
        return dict(fields={f:getattr(s,f)[slot].detach().cpu().clone() for f in STATE_FIELDS},
                    history=s._history[slot,:length].detach().cpu().clone(),length=length)

    def same_state(self,a,b):
        t=self.t
        return a['length']==b['length'] and all(t.equal(a['fields'][f].reshape(-1).view(t.uint8),b['fields'][f].reshape(-1).view(t.uint8)) for f in STATE_FIELDS) and t.equal(a['history'].view(t.uint8),b['history'].view(t.uint8))

    def token_scales(self,batch):
        t=self.t;s=self.state
        slots=batch.idx_mapping[:batch.num_reqs].cpu().tolist()
        starts=batch.query_start_loc[:batch.num_reqs+1].cpu().tolist()
        positions=batch.positions[:batch.num_tokens].cpu().tolist();tokens=batch.input_ids[:batch.num_tokens].cpu().tolist()
        expected=[];members=[]
        for i,(rid,slot) in enumerate(zip(batch.req_ids,slots)):
            if rid not in self.requests or s._dynamic_indices.get(rid)!=slot or self.runner.req_states.req_id_to_index.get(rid)!=slot:
                raise RuntimeError('Model/R/request slot disagreement')
            row=self.rows[self.requests[rid]];key=self.key(rid);a,b=starts[i:i+2]
            validate_input_slice(row['prompt_token_ids'],row['forced_token_prefix'],positions[a:b],tokens[a:b])
            if self.reference is not None:
                snap=self.reference['states'][key]
                if snap['length']!=len(row['prompt_token_ids'])+key[1]:raise RuntimeError('Reference snapshot offset mismatch')
                for f in STATE_FIELDS:getattr(s,f)[slot].copy_(snap['fields'][f])
                s._history[slot,:snap['length']].copy_(snap['history']);s._history_lengths[rid]=snap['length']
                if not self.same_state(self.snapshot(rid,slot),snap):raise RuntimeError('R snapshot restoration not byte exact')
                self.state_locks+=1
            snap=self.snapshot(rid,slot);self.before[rid]=snap
            expected.extend(snap['history'][positions[a:b]].tolist())
            self.forward_tickets[rid]=key
            members.append(dict(request_id=rid,row=self.requests[rid],count=key[1],slot=slot,
                computed=int(batch.num_computed_tokens_np[i]),scheduled=int(batch.num_scheduled_tokens[i]),
                prefill=int(batch.prefill_len_np[i]),input_positions=positions[a:b]))
        scales=self.original_scales(batch)
        if not t.equal(scales.cpu().view(t.uint8),t.tensor(expected,dtype=scales.dtype).view(t.uint8)):raise RuntimeError('Effective token scales differ from frozen history')
        self.geometry.append(dict(members=members,num_tokens=batch.num_tokens,padded_tokens=batch.num_tokens_after_padding))
        return scales

    def __call__(self,logits,batch,**kw):
        t=self.t
        if kw or batch.num_draft_tokens:raise RuntimeError('Unsupported diagnostic sampling mode')
        valid=(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np).tolist()
        slots=batch.idx_mapping[:batch.num_reqs].cpu().tolist()
        current=[]
        for i,(rid,slot,v) in enumerate(zip(batch.req_ids,slots,valid)):
            if not v:continue
            key=self.key(rid);row=self.rows[key[0]]
            if self.forward_tickets.get(rid)!=key:raise RuntimeError('Sampler ran without checked pre-forward state')
            if int(batch.seq_lens[i])!=len(row['prompt_token_ids'])+key[1]:raise RuntimeError('Forced-token clock mismatch')
            snap=self.snapshot(rid,slot)
            if not self.same_state(snap,self.before[rid]):raise RuntimeError('Controller state changed inside model forward')
            self.states[key]=snap
            if key[1] in row['capture_positions']:self.logits[key]=logits[i].detach().cpu().clone()
            current.append((i,rid,key))
        output=self.native(logits,batch)
        for i,rid,key in current:
            self.native_draws[key]=int(output.sampled_token_ids[i,0].cpu())
            output.sampled_token_ids[i,0]=self.rows[key[0]]['forced_token_prefix'][key[1]]
            self.counts[rid]+=1
        # Native model_runner.observe_sample consumes these FORCED tokens and its
        # already saved raw maximum probabilities. No controller formula is copied.
        return output

    def observe_sample(self,batch,tokens,max_probabilities):
        valid=(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np).tolist()
        for i,(rid,v) in enumerate(zip(batch.req_ids,valid)):
            if not v:continue
            key=(self.requests[rid],self.counts[rid]-1)
            if int(tokens[i,0])!=self.rows[key[0]]['forced_token_prefix'][key[1]]:raise RuntimeError('Native R did not receive the forced token')
            self.max_probabilities[key]=float(max_probabilities[i].cpu())
            if not math.isfinite(self.max_probabilities[key]):raise RuntimeError('Nonfinite native confidence')
        return self.original_observe(batch,tokens,max_probabilities)

    def finish(self):
        t=self.t;expected={(i,k) for i in range(8) for k in range(128)}
        if set(self.states)!=expected or set(self.native_draws)!=expected or set(self.max_probabilities)!=expected:raise RuntimeError('Incomplete forced prefix trace')
        if set(self.logits)!={(i,k) for i,r in enumerate(self.rows) for k in r['capture_positions']}:raise RuntimeError('Incomplete selected logits')
        report=dict(passed=True,state_positions=len(self.states),logit_positions=len(self.logits),state_locks=self.state_locks,
            native_token_differences=[],probability_differences=[],logit_differences=[])
        if self.reference is not None:
            for key in sorted(expected):
                if not self.same_state(self.states[key],self.reference['states'][key]):raise RuntimeError('Input states not identical across layouts')
                if self.native_draws[key]!=self.reference['native_draws'][key]:report['native_token_differences'].append(list(key))
                if self.max_probabilities[key]!=self.reference['max_probabilities'][key]:report['probability_differences'].append(list(key))
            for key,x in self.logits.items():
                ref=self.reference['logits'][key];exact=t.equal(x.contiguous().view(t.uint8),ref.contiguous().view(t.uint8))
                if not exact:
                    delta=x.float()-ref.float()
                    report['logit_differences'].append(dict(position=list(key),max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/ref.float().norm().clamp_min(1e-30))))
            report['repeat_exact']=not any(report[k] for k in ('native_token_differences','probability_differences','logit_differences'))
            report['same_logical_geometry']=logical_geometry(self.geometry)==logical_geometry(self.reference['geometry'])
        t.save(dict(states=self.states,logits=self.logits,native_draws=self.native_draws,max_probabilities=self.max_probabilities),self.folder/'trace.pt')
        (self.folder/'geometry.json').write_text(json.dumps(self.geometry),encoding='utf8')
        (self.folder/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        if self.strict_repeat and not (report['repeat_exact'] and report['same_logical_geometry']):raise RuntimeError('Same-layout repeat differs; stop before layout comparisons')
        return report

    def close(self):
        self.runner.sampler=self.native;self.state.token_scales=self.original_scales
        self.state.observe_sample=self.original_observe
        self.core.scheduler._preempt_request=self.original_preempt
