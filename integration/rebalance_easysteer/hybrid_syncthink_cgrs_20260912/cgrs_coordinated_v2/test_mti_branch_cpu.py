"""Actual causal attention toy checks, not an LLM correctness test."""
import json
from types import SimpleNamespace
import torch
from mti_branch import isolated_cue, scratch_plan


class Cache:
    def __init__(self, keys): self.keys = keys
    def get_seq_length(self): return self.keys.shape[1]


class Decoder:
    training = False
    fail = False
    def __call__(self, input_ids, past_key_values, position_ids, **kwargs):
        # Distinct position-dependent features make stale/incorrect positions observable.
        new = torch.stack((input_ids.float(), position_ids.float(),
                           torch.ones_like(input_ids).float()), -1) / 20
        old = past_key_values.keys
        all_keys = torch.cat((old, new), 1)
        past_key_values.keys = all_keys
        torch.rand(3)  # prove even a stochastic callback cannot advance caller RNG
        if self.fail: raise RuntimeError('injected after cache mutation')
        q = new.unsqueeze(1); k = all_keys.unsqueeze(1)
        mask = torch.arange(k.shape[-2])[None, :] <= position_ids[0, :, None]
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, k,
                  attn_mask=mask, dropout_p=0).squeeze(1)
        return SimpleNamespace(last_hidden_state=hidden)


def run():
    checks=[]
    for length in (15, 16, 17, 31, 32, 33):
        for cue_length in (1, 2, 17):
            p = scratch_plan(length, cue_length, 16)
            assert p.shared_full_blocks*16+p.copy_tail_tokens == length
            assert p.scratch_blocks*16 >= p.copy_tail_tokens+cue_length
            assert (p.scratch_blocks-1)*16 < p.copy_tail_tokens+cue_length
            assert p.cue_positions == tuple(range(length, length+cue_length))
    checks.append('18_block_boundary_and_multiblock_plans')
    d=Decoder(); ids=torch.tensor([[2, 5, 7, 11]])
    positions=torch.arange(4).unsqueeze(0)
    keys=torch.stack((ids.float(),positions.float(),torch.ones_like(ids).float()),-1)/20
    cache=Cache(keys.clone()); original=keys.clone(); rng=torch.get_rng_state().clone()
    cue=torch.tensor([[17, 23]])
    hidden=isolated_cue(d,cache,4,cue)
    assert torch.equal(cache.keys,original) and cache.get_seq_length()==4
    assert torch.equal(torch.get_rng_state(),rng)
    checks.append('parent_cache_and_rng_unchanged')
    full=torch.cat((ids,cue),1)
    ref=d(input_ids=full,past_key_values=Cache(torch.empty(1,0,3)),
          position_ids=torch.arange(6).unsqueeze(0)).last_hidden_state[:,-1:]
    torch.testing.assert_close(hidden,ref,atol=1e-6,rtol=1e-6)
    checks.append('cue_self_attention_matches_full_prefix')
    first=isolated_cue(d,cache,4,cue[:,:1])
    assert not torch.allclose(first,hidden)
    checks.append('multi_token_cue_not_silently_single_token')
    d.fail=True; rng=torch.get_rng_state().clone()
    try: isolated_cue(d,cache,4,cue)
    except RuntimeError as exc: assert 'injected' in str(exc)
    else: raise AssertionError('failure lost')
    assert torch.equal(cache.keys,original) and torch.equal(torch.get_rng_state(),rng)
    checks.append('exception_preserves_cache_and_rng')
    for n in (0,3,5):
        try: isolated_cue(d,cache,n,cue)
        except ValueError: pass
        else: raise AssertionError('bad clock accepted')
    checks.append('cache_clock_mismatch_rejected')
    return dict(checks_passed=checks,gpu_used=False,
                limit='Toy single-layer attention; no HF/vLLM or steering equivalence claim')


if __name__=='__main__': print(json.dumps(run(),indent=2))
