"""Instrument ABI checks only. Actual L27 histories are checked on the GPU."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from feedback_replay_capture import FeedbackCapture


def run():
    class Native:
        def add_request(self, value):
            return ('registered', value)
        def returns_logprobs(self, _):
            return False
        def __call__(self, logits, batch, **kw):
            return NS(sampled_token_ids=logits.argmax(-1)[:, None])
    native = Native()
    state = NS(_coefs=torch.tensor([-1., .1]), _prev_step_mean=torch.tensor([.8, .9]), _in_think=torch.tensor([True, True]))
    owner = NS(original_sampler=native, count=torch.zeros(2, dtype=torch.long),
        lex_open=torch.tensor([True, False]), thinking=torch.ones(2, dtype=torch.bool),
        lex_state=torch.tensor([3, 4]))
    def l27(logits, batch, **kw):
        logits[:, 2] -= .6931471805599453
        result = owner.original_sampler(logits, batch, **kw)
        owner.count[batch.idx_mapping.long()] += torch.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens >= batch.prefill_len_np).long()
        return result
    # Simulate the native sampler's request-management ABI through the wrapper.
    l27.add_request = native.add_request
    runner = NS(max_num_reqs=2, device='cpu', sampler=l27, steer_vector_state=state)
    owner.runner = runner
    c = FeedbackCapture(owner, 2)
    c.register(0, [1, 2]); c.register(1, [2, 1]); c.install()
    assert runner.sampler.add_request('test') == ('registered', 'test')
    batch = NS(num_reqs=2, idx_mapping=torch.tensor([1, 0]), idx_mapping_np=np.array([1, 0]),
        num_computed_tokens_np=np.array([4, 4]), num_scheduled_tokens=np.array([1, 1]), prefill_len_np=np.array([5, 5]))
    raw = torch.tensor([[1., 2., 3.], [3., 1., 2.]])
    output = runner.sampler(raw.clone(), batch)
    assert output.sampled_token_ids[:, 0].tolist() == [2, 1]
    expected = raw.clone(); expected[:, 2] -= .6931471805599453
    values = expected.log_softmax(-1)[[0, 1], [2, 1]]
    assert torch.allclose(c.traces['logp'][[1, 0], 0], values)
    assert torch.allclose(c.traces['rawmax'][[1, 0], 0], raw.softmax(-1).max(-1).values)
    assert c.traces['lex_gate'][:, 0].tolist() == [1, 0]
    assert c.traces['coefficient_before'][:, 0].tolist()[0] == -1
    c.register(0, [0]); owner.count[0] = 0
    assert torch.isnan(c.traces['logp'][0]).all() and c.length[0] == 1
    c.close()
    assert runner.sampler is l27 and owner.original_sampler is native
    return dict(status='CPU instrumentation checks passed', checks=['forced token matches request slot permutation',
        'scores use post-penalty full distribution', 'raw confidence retained before penalty',
        'negative finite-mean lexical gate', 'slot buffer reset', 'request-management ABI forwarded', 'hooks restored'],
        scope='No model forward, no vLLM-equivalence or efficacy claim')


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
