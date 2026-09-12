"""CPU tests of the actual runtime state: no model or CUDA allocation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT/'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/sample/hybrid_termination.py'
spec = importlib.util.spec_from_file_location('hybrid_under_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def batch(ids, slots, lengths, prefill=None):
    return SimpleNamespace(req_ids=ids, num_reqs=len(ids), num_draft_tokens=0,
                           idx_mapping=torch.tensor(slots), seq_lens=torch.tensor(lengths),
                           num_computed_tokens_np=np.array(lengths)-1,
                           num_scheduled_tokens=np.ones(len(ids), dtype=int),
                           prefill_len_np=np.array(prefill if prefill else [1]*len(ids)))


def main():
    config = dict(mode='enforce', entropy_weight=.8, pacing_cap=64, end_token_id=151649)
    state = module.HybridTerminationState(3, 'cpu')
    state.add_request('a', 2, [151648], 1, config)
    logits = torch.full((1,151650), -40.)
    logits[0,151649] = 5
    before = logits.clone()
    ticket = state.read_apply(logits, batch(['a'],[2],[1]))
    assert torch.isfinite(logits).sum() == 1 and logits[0,151649] == 0
    state.observe(ticket, torch.tensor([[151649]]))
    assert state.count[2] == 1 and state.closed[2]
    next_logits = before.clone()
    ticket = state.read_apply(next_logits, batch(['a'],[2],[2]))
    assert torch.equal(next_logits,before)
    state.observe(ticket, torch.tensor([[151648]]))
    assert state.closed[2]  # A later think token cannot reopen termination.
    state.remove_request('a')
    state.add_request('b',2,[151648],1,dict(config,mode='shadow'))
    same = before.clone()
    ticket = state.read_apply(same,batch(['b'],[2],[1]))
    assert torch.equal(same,before) and state.count[2] == 0
    state.observe(ticket,torch.tensor([[7]]))
    assert state.count[2] == 1 and not state.closed[2]
    partial = before.clone()
    assert state.read_apply(partial,batch(['b'],[2],[1],[3])) is None
    assert state.count[2] == 1
    try:
        state.read_apply(partial,batch(['b'],[2],[1]))
        raise AssertionError('Duplicate/replayed accepted position permitted')
    except RuntimeError:
        pass
    # Invalid support and invalid resume are failures, never silent resets.
    try:
        state.add_request('c',0,[151648],2,config)
        raise AssertionError('Unvalidated generated prefix accepted')
    except ValueError:
        pass
    bad = before.clone(); bad[0,0] = torch.nan
    try:
        state.read_apply(bad,batch(['b'],[2],[2]))
        raise AssertionError('NaN accepted')
    except RuntimeError:
        pass
    off_runner = SimpleNamespace()
    req = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={module.NAMESPACE:{'mode':'off'}}))
    module.admit(off_runner,req,0)
    assert not hasattr(off_runner,'hybrid_termination')
    result = state.finish_all()
    assert result['requests']['a']['end_position'] == 0
    assert result['requests']['b']['accepted_tokens'] == 1
    print(json.dumps(dict(status='pass',device='cpu',model_forwards=0,
                         checks=['forced support','sticky close','slot reset','shadow identity',
                                 'partial prefill ignored','duplicate clock rejected','resume rejected',
                                 'NaN rejected','off no allocation','request receipts'])))


if __name__ == '__main__':
    main()
