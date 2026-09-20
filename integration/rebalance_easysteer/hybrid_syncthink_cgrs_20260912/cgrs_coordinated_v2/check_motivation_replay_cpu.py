"""CPU checks for probability preservation, slot clocks, padding and reuse."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from motivation_replay_sampler import ReplayBuffers, MotivationReplaySampler


def main():
    b = ReplayBuffers(3, 4, 'cpu')
    b.register(2, [1, 0], 5)
    b.register(0, [2], 8)
    logits = torch.tensor([[2., 1., 0.], [0., 3., 1.]])
    saved = logits.clone()
    idx = torch.tensor([2, 0])
    class Native:
        def returns_logprobs(self, _): return False
        def __call__(self, logits, batch):
            return SimpleNamespace(sampled_token_ids=logits.argmax(-1)[:, None].int())
    sampler = MotivationReplaySampler(Native(), b)
    batch = SimpleNamespace(num_draft_tokens=0, num_reqs=2, idx_mapping=idx,
        idx_mapping_np=np.array([2, 0]), num_computed_tokens_np=np.array([4, 2]),
        num_scheduled_tokens=np.array([1, 1]), prefill_len_np=np.array([5, 8]),
        seq_lens=torch.tensor([5, 3]))
    output = sampler(logits, batch)
    assert output.sampled_token_ids[:, 0].tolist() == [1, 1]
    assert torch.equal(saved, logits) and b.count.tolist() == [0, 0, 1]
    assert torch.allclose(b.logmax[2, 0], torch.log_softmax(saved, -1)[0].max())
    batch.num_computed_tokens_np = np.array([5, 7])
    batch.seq_lens = torch.tensor([6, 8])
    output = sampler(logits, batch)
    assert output.sampled_token_ids[:, 0].tolist() == [0, 2]
    assert len(b.completed(2)) == 2 and len(b.completed(0)) == 1
    b.register(2, [2], 12)
    assert int(b.count[2]) == 0 and torch.isnan(b.logmax[2]).all()
    try:
        b.capture(logits[:1], torch.tensor([2]), torch.tensor([True]), torch.tensor([13]))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Misaligned prefix accepted')
    print(json.dumps(dict(passed=True, checks=['raw_logits_unchanged', 'full_vocab_logmax',
        'forced_tokens', 'invalid_prefill_not_counted', 'slot_permutation', 'slot_reset',
        'prefix_mismatch_rejected'], gpu_validated=False)))


if __name__ == '__main__': main()
