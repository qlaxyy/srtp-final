"""Opt-in, same-logits sampler checks; no model/controller implementation fork."""
import copy
import hashlib
import json
from types import SimpleNamespace

from adapter import Sampler
from policy import PENALTY

FIELDS=('opening','thinking','count','prompt_len','eligible_count','changed_count',
        'first_change','first_reflection')
PATHS=('native_repeat','shadow_history','inactive_rc14','inactive_history','positive_rc14')


def capture_positions(first_difference, cap=128):
    if type(first_difference) is not int or not 0 <= first_difference < cap:
        raise ValueError('Difference position outside fixed diagnostic window')
    return sorted({0,max(0,first_difference-1),first_difference,cap-1})


def verify_mapping(active, model_slots, controller_slots, batch_slots):
    reverse={slot:rid for rid,slot in active.items()}
    if len(reverse)!=len(active) or len(batch_slots)!=len(set(batch_slots)):
        raise RuntimeError('Duplicate active or batch slot')
    result=[]
    for slot in batch_slots:
        rid=reverse.get(slot)
        if rid is None or model_slots.get(rid)!=slot or controller_slots.get(rid)!=slot:
            raise RuntimeError('Model/controller/adapter slot disagreement')
        result.append(rid)
    return result


def clone_owner(owner,native):
    cloned=copy.copy(owner)
    for key in FIELDS:setattr(cloned,key,getattr(owner,key).clone())
    cloned.original_sampler=native
    return cloned


def input_variant(owner,native,kind):
    cloned=clone_owner(owner,native)
    if kind=='shadow_history':cloned.mode='shadow'
    elif kind=='inactive_rc14':
        cloned.mode='negative';cloned.history_gate='none';cloned.opening.zero_()
    elif kind=='inactive_history':
        cloned.mode='negative';cloned.first_reflection.fill_(-1)
    elif kind=='positive_rc14':
        # Synthetic active gate tests the mask/penalty, not a natural R decision.
        cloned.mode='negative';cloned.history_gate='none'
        cloned.opening.fill_(True);cloned.thinking.fill_(True)
        s=owner.runner.steer_vector_state
        cloned.runner=SimpleNamespace(steer_vector_state=SimpleNamespace(
            _coefs=owner.torch.full_like(s._coefs,-1),
            _prev_step_mean=owner.torch.ones_like(s._prev_step_mean)))
    else:raise ValueError(kind)
    return cloned


def check_case(owner,native,raw,batch,kind,baseline):
    """Compare pre-native logits and draws; no mutable owner state is shared."""
    torch=owner.torch;seen={}
    def capture(x,b,**kw):
        seen['logits']=x.clone()
        return native(x,b,**kw)
    x=raw.clone()
    if kind=='native_repeat':output=capture(x,batch)
    else:output=Sampler(input_variant(owner,capture,kind))(x,batch)
    expected=raw.clone()
    if kind=='positive_rc14':
        valid=torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=raw.device)
        values=expected[:,owner.ids]
        expected[:,owner.ids]=torch.where(valid[:,None],values-PENALTY,values)
        reference=native(expected.clone(),batch).sampled_token_ids.clone()
    else:reference=baseline
    # BF16 raw logits are finite; byte view also detects signed-zero changes.
    exact_logits=torch.equal(seen['logits'].contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
    exact_tokens=torch.equal(output.sampled_token_ids,reference)
    return dict(path=kind,logits_equal=exact_logits,tokens_equal=exact_tokens,
                sampled_token_ids=output.sampled_token_ids.detach().cpu().tolist(),
                passed=exact_logits and exact_tokens)


class SamplerDiagnostic:
    def __init__(self,owner,targets,folder,max_capture_bytes):
        self.owner=owner;self.native=owner.original_sampler;self.targets=targets
        self.folder=folder;self.max_capture_bytes=max_capture_bytes;self.bytes=0
        self.seen=set();self.calls=0;self.checks=0

    def __getattr__(self,name):return getattr(self.native,name)

    def __call__(self,logits,batch,**kw):
        if kw:raise RuntimeError('Diagnostic supports unmodified native sampling only')
        o=self.owner;t=o.torch;self.calls+=1
        slots=batch.idx_mapping[:batch.num_reqs].detach().cpu().tolist()
        rids=verify_mapping(o.active,o.runner.req_states.req_id_to_index,
            o.runner.steer_vector_state._dynamic_indices,slots)
        counts=o.count[slots].detach().cpu().tolist()
        valid=(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np).tolist()
        hits=[(rid,count) for rid,count,v in zip(rids,counts,valid)
              if v and count in self.targets[rid] and (rid,count) not in self.seen]
        if not hits:return self.native(logits,batch)
        raw=logits.clone();state={k:getattr(o,k).clone() for k in FIELDS}
        coefs=o.runner.steer_vector_state._coefs.clone()
        mean=o.runner.steer_vector_state._prev_step_mean.clone()
        seeds=self.native.sampling_states.seeds.gpu.clone()
        positions=batch.positions[batch.logits_indices].clone()
        blob=dict(logits=raw.cpu(),owner={k:v.cpu() for k,v in state.items()},coefs=coefs.cpu(),
            prev_step_mean=mean.cpu(),seed=seeds.cpu(),positions=positions.cpu(),
            slots=slots,requests=rids,hits=hits,seq_lens=batch.seq_lens.cpu(),
            num_computed=batch.num_computed_tokens_np.tolist(),num_scheduled=batch.num_scheduled_tokens.tolist(),
            prefill_len=batch.prefill_len_np.tolist(),valid=valid)
        estimated=sum(x.numel()*x.element_size() for x in (raw,coefs,mean,seeds,positions,*state.values()))
        if self.bytes+estimated>self.max_capture_bytes:raise RuntimeError('Capture byte ceiling')
        self.bytes+=estimated
        path=self.folder/f'capture_{self.checks:03d}.pt'
        t.save(blob,path)
        report=dict(capture=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hits=hits,
            mapping=dict(zip(rids,slots)),cases=[],owner_unchanged=False)
        output=self.native(logits,batch)
        baseline=output.sampled_token_ids.clone()
        report['native_sampled_token_ids']=baseline.cpu().tolist()
        sampled=output.num_sampled.clone();rejected=output.num_rejected.clone()
        try:
            for kind in PATHS:report['cases'].append(check_case(o,self.native,raw,batch,kind,baseline))
            report['owner_unchanged']=all(t.equal(getattr(o,k).view(t.uint8),v.view(t.uint8)) for k,v in state.items())
            report['R_unchanged']=t.equal(coefs.view(t.uint8),o.runner.steer_vector_state._coefs.view(t.uint8)) and t.equal(mean.view(t.uint8),o.runner.steer_vector_state._prev_step_mean.view(t.uint8))
            report['seed_unchanged']=t.equal(seeds,self.native.sampling_states.seeds.gpu)
            report['live_output_unchanged']=t.equal(output.sampled_token_ids,baseline) and t.equal(output.num_sampled,sampled) and t.equal(output.num_rejected,rejected)
            report['passed']=all(c['passed'] for c in report['cases']) and all(report[k] for k in ('owner_unchanged','R_unchanged','seed_unchanged','live_output_unchanged'))
        finally:
            with (self.folder/'checks.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(report)+'\n');f.flush()
        if not report.get('passed'):raise RuntimeError('Same-input sampler invariant failed')
        self.seen.update(hits);self.checks+=1
        return output
