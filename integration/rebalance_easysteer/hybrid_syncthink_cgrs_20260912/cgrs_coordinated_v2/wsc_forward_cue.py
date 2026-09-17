"""Bounded synchronous engineering adapter; CPU boundary scoring, no rollback.

Wrap the base sampler inside RC14, so all downstream state sees accepted IDs.
The disabled path installs nothing. This is not a production speed path.
"""
import time
import numpy as np
from wsc_feature_contract_cpu import BoundaryLedger
from wsc_recovery_policy_cpu import LoopState
from wsc_shadow import WSCShadow


class CueState:
    def __init__(self, cue, cap=16000, active=True):
        if not 1 <= len(cue) <= 64 or cap < 1:
            raise ValueError('Invalid cue or budget')
        if any(x in (151643,151645,151648,151649) or x < 0 for x in cue):
            raise ValueError('Control token in cue')
        self.cue=tuple(cue);self.cap=cap;self.active=active
        self.loop=LoopState();self.offset=None;self.first=None
        self.skipped_budget=False;self.forced=[]

    def observe(self, score, chunk, generated):
        fired=self.loop.observe(score,chunk)
        if fired:
            if self.cap-generated < len(self.cue):self.skipped_budget=True
            elif self.active:self.offset=0;self.first=generated
        return fired

    def next_token(self, generated):
        if self.offset is None or self.offset==len(self.cue):return None
        if generated>=self.cap:raise RuntimeError('Cue exceeds total cap')
        token=self.cue[self.offset];self.offset+=1
        self.forced.append(dict(position=generated,token=token))
        return token


class ForcedSampler:
    def __init__(self, owner, original):self.owner=owner;self.original=original
    def __getattr__(self, name):return getattr(self.original,name)
    def __call__(self, logits, batch, **kwargs):
        result=self.original(logits,batch,**kwargs)
        if result.sampled_token_ids.shape!=(batch.num_reqs,1):raise ValueError('Sample shape')
        for row,slot in enumerate(self.owner.current_slots):
            state=self.owner.states[slot]
            token=state.next_token(len(self.owner.ledgers[slot].accepted))
            if token is not None:result.sampled_token_ids[row,0]=token
        return result


class WSCForwardCue(WSCShadow):
    def __init__(self, adapter, weight, bias, boundaries, cue, *, enabled=False, active=True, cap=16000):
        if not enabled:self.enabled=False;return
        self.adapter=adapter;self.weight=np.asarray(weight,dtype=np.float32).reshape(-1)
        self.bias=float(bias)
        if self.weight.shape!=(1536,) or not np.isfinite(self.weight).all() or not np.isfinite(self.bias):
            raise ValueError('Invalid public probe')
        self.boundaries=set(boundaries);self.cue=tuple(cue);self.active_cue=active;self.cap=cap
        self.ledgers={};self.states={};self.events={};self.current_slots=[]
        self.score_seconds=0.;self.host_seconds=0.
        super().__init__(adapter.runner,enabled=True,max_calls=cap)
        self.base_sampler=adapter.original_sampler
        self.forcer=ForcedSampler(self,self.base_sampler)
        adapter.original_sampler=self.forcer

    def __call__(self, logits, batch, **kwargs):
        start=time.monotonic();frame=self.pending
        if frame is None or not frame.get('checked'):raise RuntimeError('Missing native feature')
        # Fixed small engineering run: synchronous transfers are explicit overhead.
        take=batch.logits_indices.long()
        slots=batch.idx_mapping[:batch.num_reqs].long().cpu().tolist()
        positions=frame['positions'][take].cpu().tolist()
        inputs=frame['ids'][take].cpu().tolist()
        if not np.all(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np):
            raise RuntimeError('Partial prefill unsupported')
        self.current_slots=slots
        for j,(slot,pos,token) in enumerate(zip(slots,positions,inputs)):
            if slot not in self.states:
                plen=int(self.adapter.prompt_len[slot].item())
                self.ledgers[slot]=BoundaryLedger(plen,1536,self.boundaries)
                self.states[slot]=CueState(self.cue,self.cap,self.active_cue);self.events[slot]=[]
            ledger=self.ledgers[slot]
            # A feature is needed only for a generated boundary, not every token.
            h=frame['hidden'][take[j]].float().cpu().numpy() if token in self.boundaries and pos>=ledger.prompt_tokens else None
            event=ledger.observe(pos,token,h)
            if event is not None:
                began=time.monotonic()
                score=float(1/(1+np.exp(-np.clip(float(event['hidden']@self.weight+self.bias),-80,80))))
                fired=self.states[slot].observe(score,event['chunk_tokens'],len(ledger.accepted))
                self.score_seconds+=time.monotonic()-began
                self.events[slot].append(dict(position=event['position'],chunk_tokens=event['chunk_tokens'],score=score,would_trigger=fired))
            if ledger.disabled:raise RuntimeError('Nonfinite native feature')
        self.host_seconds+=time.monotonic()-start
        result=super().__call__(logits,batch,**kwargs)
        began=time.monotonic()
        for slot,token in zip(slots,result.sampled_token_ids[:,0].cpu().tolist()):self.ledgers[slot].accept(token)
        self.host_seconds+=time.monotonic()-began
        return result

    def report(self):
        return dict(score_seconds=self.score_seconds,host_observation_seconds=self.host_seconds,
                    implementation='CPU boundary scoring with synchronous metadata transfers; engineering only',
                    slots={str(s):dict(events=self.events[s],first_forced=state.first,forced=state.forced,
                                      skipped_budget=state.skipped_budget) for s,state in self.states.items()})

    def close(self):
        if not self.enabled:return
        if self.adapter.original_sampler is not self.forcer:raise RuntimeError('Base sampler replaced unexpectedly')
        self.adapter.original_sampler=self.base_sampler
        super().close()
