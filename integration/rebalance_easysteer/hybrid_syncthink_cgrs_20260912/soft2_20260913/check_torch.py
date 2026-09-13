"""Run in the EXISTING inference Python, CPU only, after isolated patch apply."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import check_runtime as existing


def main():
    # Existing actual-state suite includes replay rejection and request reset.
    for dtype in (torch.float32, torch.bfloat16):
        existing.check_dtype(dtype)
        state = existing.module.HybridTerminationState(3, 'cpu')
        cfg = dict(mode='soft', entropy_weight=.8, pacing_cap=64, end_token_id=151649)
        state.add_request('soft', 2, [151648], 1, cfg)
        state.add_request('shadow', 0, [151648], 1, dict(cfg, mode='shadow'))
        raw = torch.full((2, 151650), -40., dtype=dtype)
        raw[:, 151649] = 5
        before = raw.clone()
        ticket = state.read_apply(raw, existing.batch(['soft','shadow'],[2,0],[1,1]))
        assert torch.equal(raw, before), 'Soft must not mutate raw R statistics'
        filtered = torch.full_like(raw, -torch.inf)
        filtered[:, 151649] = -1
        filtered[:, 7] = 0
        before = filtered.clone()
        state.apply_after_filter(filtered)
        assert torch.equal(torch.isfinite(filtered), torch.isfinite(before))
        assert torch.equal(filtered[:, :151649], before[:, :151649])
        assert torch.equal(filtered[1], before[1])
        assert abs(float(filtered[0,151649] + 1)-math.log(2)) < .004
        assert int(state.bias_count[2]) == 1 and int(state.first_bias[2]) == 0
        again = filtered.clone(); state.apply_after_filter(filtered)
        assert torch.equal(again,filtered), 'Do not reuse previous sampling ticket'
        state.observe(ticket, torch.tensor([[7],[7]]))
        # Reordered batch: a filtered-out end remains filtered, no forced insert.
        raw = before.clone(); raw[:, 151649] = 5
        ticket = state.read_apply(raw, existing.batch(['shadow','soft'],[0,2],[2,2]))
        filtered = before.clone(); filtered[1,151649] = -torch.inf
        state.apply_after_filter(filtered)
        assert torch.isneginf(filtered[1,151649])
        assert int(state.filtered_trigger_count[2]) == 1
        state.observe(ticket, torch.tensor([[7],[151649]]))
        # Closed thinking stays closed; slot reuse clears every new counter.
        raw = before.clone(); raw[:,151649] = 5
        state.read_apply(raw, existing.batch(['shadow','soft'],[0,2],[3,3]))
        filtered = before.clone(); state.apply_after_filter(filtered)
        assert torch.equal(filtered, before)
        state.remove_request('soft')
        state.add_request('reuse',2,[151648],1,cfg)
        assert int(state.bias_count[2]) == 0 and int(state.first_bias[2]) == -1
        assert int(state.filtered_trigger_count[2]) == 0 and not state.closed[2]
        print(json.dumps(dict(status='pass', dtype=str(dtype), device='cpu',
                              extra_forwards=0, cases='soft support, mixed rows, reorder, ticket, reset, close')))


if __name__ == '__main__':
    main()
