"""Reference contract for processed-token WSC observations; no live hooks.

Chosen from pinned public runtime and layer27 reproduction config, not inferred
from checkpoint dimensions or picked using benchmark scores. Both project Qwen
models have28blocks: HF hidden index27 is raw decoder26 output, before block27.
vLLM represents this boundary as a hidden/residual pair; neither tensor alone is
the HF residual-stream feature. Native BF16 equivalence remains a GPU gate.
"""
import numpy as np


class BoundaryLedger:
    def __init__(self,prompt_tokens,dimension,boundary_ids):
        if prompt_tokens<1 or dimension not in (1536,3584):raise ValueError('Unsupported model contract')
        self.prompt_tokens=prompt_tokens;self.dimension=dimension;self.boundaries=set(boundary_ids)
        self.accepted=[];self.observed=set();self.last_boundary=-1
        self.closed=False;self.disabled=False

    def accept(self,token):
        self.accepted.append(int(token))

    def observe(self,absolute_input_position,input_token,hidden,hf_index=27):
        if hf_index!=27:raise ValueError('Layer drift: expected fixedHF27')
        if self.disabled or self.closed:return None
        relative=absolute_input_position-self.prompt_tokens
        if relative<0:return None # prompt prefill never counts as generated thought
        if relative>=len(self.accepted) or self.accepted[relative]!=input_token:
            raise ValueError('Input must be the actually accepted processed token, not the newly predicted token')
        if input_token==151649:self.closed=True;return None
        if input_token not in self.boundaries:return None
        if relative in self.observed:raise ValueError('Duplicate boundary observation')
        if relative<=self.last_boundary:raise ValueError('Out-of-order boundary')
        vector=np.asarray(hidden)
        if vector.shape!=(self.dimension,) or not np.isfinite(vector).all():
            self.disabled=True;return None
        chunk_tokens=relative-self.last_boundary-1
        self.observed.add(relative);self.last_boundary=relative
        return dict(position=relative,chunk_tokens=chunk_tokens,hidden=vector.astype(np.float32,copy=True))


def self_test():
    p=BoundaryLedger(10,1536,{271});h=np.ones(1536,dtype=np.float32)
    assert p.observe(9,99,h) is None
    p.accept(100)
    assert p.observe(10,100,h) is None
    try:p.observe(11,271,h)
    except ValueError:pass
    else:raise AssertionError('Unaccepted future delimiter was scored')
    p.accept(271);row=p.observe(11,271,h);assert row['chunk_tokens']==1
    h[:]=0;assert row['hidden'].sum()==1536 # buffer mutations cannot rewrite evidence
    try:p.observe(11,271,h)
    except ValueError:pass
    else:raise AssertionError('Duplicate accepted')
    try:p.observe(11,271,h,hf_index=28)
    except ValueError:pass
    else:raise AssertionError('Layer drift accepted')
    p.accept(151649);assert p.observe(12,151649,h) is None and p.closed
    q=BoundaryLedger(10,1536,{271});q.accept(271);h[0]=np.nan
    assert q.observe(10,271,h) is None and q.disabled
    r=BoundaryLedger(10,1536,{271});assert not r.accepted and not r.observed
    return dict(prefill_excluded=True,future_token_rejected=True,accepted_boundary_position=True,
        evidence_copy=True,duplicate_rejected=True,fixed_layer_enforced=True,answer_region_closed=True,
        nonfinite_disables_request=True,request_reset=True,gpu_used=False)


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
