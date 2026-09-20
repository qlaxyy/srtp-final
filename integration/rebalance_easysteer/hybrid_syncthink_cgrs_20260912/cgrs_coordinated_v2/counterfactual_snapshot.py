"""Explicit worker-side fork staging; no hook installed merely by importing.

Capture after an accepted boundary, before that boundary is forwarded. Native
replay owns KV reconstruction. Skip changes one historical injection only,
not the dynamic coefficient used by the lexical sampler or future updates.
The scheduler must separately present the original prompt and generated IDs.
"""
import copy
import hashlib
import json

FIELDS=('_coefs','_step_prob_sum','_step_tok_count','_prev_step_mean',
        '_in_think','_paper_strength','_paper_pending')

def identity(prompt,generated):
    return hashlib.sha256(json.dumps([prompt,generated],separators=(',',':')).encode()).hexdigest()

def capture(adapter,rid,prompt_ids,generated_ids,*,asset_identity):
    if not generated_ids:raise ValueError('A generated boundary is required')
    if not asset_identity:raise ValueError('Asset identity is required')
    if adapter.runner.vllm_config.scheduler_config.async_scheduling:
        raise ValueError('Fork replay requires synchronous execution')
    s=adapter.runner.steer_vector_state;t=adapter.torch
    if not s.supports_kv_replay:raise ValueError('Native history required')
    if rid not in adapter.active or rid not in s._dynamic_indices:raise ValueError('Inactive parent')
    slot=adapter.active[rid];idx=s._dynamic_indices[rid]
    if slot!=idx:raise ValueError('Lexical/native slot mismatch')
    params=s._dynamic_params[rid]
    if generated_ids[-1] not in params.boundary_token_ids:raise ValueError('Not a native boundary')
    n=len(prompt_ids)+len(generated_ids)
    if s._history_lengths[rid]!=n or s._prompt_lengths[rid]!=len(prompt_ids):raise ValueError('Native clock mismatch')
    lexical={k:int(getattr(adapter,k)[slot].item()) for k in adapter.fields}
    if lexical['count']!=len(generated_ids) or lexical['prompt_len']!=len(prompt_ids):raise ValueError('Lexical clock mismatch')
    values=[getattr(s,k)[idx].detach().cpu().clone() for k in FIELDS]
    if len(s._state_fields())!=len(FIELDS):raise ValueError('Unknown native field schema')
    for native,expected in zip(s._state_fields(),FIELDS):
        if native is not getattr(s,expected):raise ValueError('Native field order changed')
    if not bool(values[4]) or int(values[2])!=0:raise ValueError('Boundary must be closed inside thinking')
    if not t.isfinite(values[0]) or not t.isfinite(values[3]):raise ValueError('Nonfinite completed state')
    history=s._history[idx,:n].detach().cpu().clone()
    if not t.isfinite(history).all():raise ValueError('Nonfinite history')
    if history[-1].item()!=values[0].item():raise ValueError('Boundary scale/history mismatch')
    return dict(version=1,asset_identity=copy.deepcopy(asset_identity),prefix_sha256=identity(prompt_ids,generated_ids),
        prompt_ids=list(prompt_ids),generated_ids=list(generated_ids),lexical=lexical,
        native=dict(params=copy.deepcopy(params),prompt_length=len(prompt_ids),history=history,fields=values))

def fork(snapshot,*,action):
    if action not in ('apply','skip'):raise ValueError(action)
    result=copy.deepcopy(snapshot);result['action']=action
    if action=='skip':result['native']['history'][-1]=0
    # Preserve native _coefs. Otherwise lexical suppression and every input
    # before the next completed step would also change, violating one action.
    return result

def stage(adapter,rid,snapshot,request,*,asset_identity):
    """Stage before native/lexical add_requests; reject ordinary long prompts.

    This does not enqueue, copy KV, or certify scheduler replay. Caller must
    arrange a genuine generated-prefix NewRequestData with num_computed=0.
    """
    s=adapter.runner.steer_vector_state
    if snapshot.get('version')!=1 or snapshot.get('action') not in ('apply','skip'):raise ValueError('Invalid fork schema')
    if snapshot['asset_identity']!=asset_identity:raise ValueError('Assets differ')
    if adapter.runner.vllm_config.scheduler_config.async_scheduling:raise ValueError('Async replay not supported')
    if request.req_id!=rid or request.num_computed_tokens!=0:raise ValueError('Invalid replay request')
    prompt=list(request.prompt_token_ids);prefix=list(request.prefill_token_ids)
    if prefix[:len(prompt)]!=prompt:raise ValueError('Prefill does not start with prompt')
    generated=prefix[len(prompt):]
    if identity(prompt,generated)!=snapshot['prefix_sha256']:raise ValueError('Prefix identity differs')
    if rid in adapter.active or rid in adapter.completed or rid in adapter.suspended or rid in s._suspended or rid in s._dynamic_indices:
        raise ValueError('Target request already exists')
    # All checks precede mutation; each child receives separate tensor storage.
    native=copy.deepcopy(snapshot['native']);lexical=copy.deepcopy(snapshot['lexical'])
    s._suspended[rid]=native;adapter.suspended[rid]=lexical

def discard_staged(adapter,rid):
    if rid in adapter.active or rid in adapter.runner.steer_vector_state._dynamic_indices:
        raise ValueError('Cannot discard an active request')
    adapter.suspended.pop(rid,None);adapter.runner.steer_vector_state._suspended.pop(rid,None)
