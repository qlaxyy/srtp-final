"""Run-isolated CPU copies for the BCC diagnostic; no global framework changes."""
import numpy as np


class CaptureStore:
    def __init__(self):
        self.run_id = None
        self.values = {}
        self.seen_runs = set()

    def begin(self, run_id):
        if self.run_id is not None or not run_id or run_id in self.seen_runs:
            raise ValueError('Active, empty or reused run ID')
        self.run_id = run_id
        self.seen_runs.add(run_id)
        self.values = {}

    def put(self, run_id, name, value, positions=None):
        if run_id != self.run_id or self.run_id is None or name in self.values:
            raise ValueError('Stale run or duplicate capture')
        tensor = value[0] if isinstance(value, tuple) else value
        if positions is not None:
            if len(tensor.shape) != 3 or tensor.shape[0] != 1:
                raise ValueError('Expected batch-one sequence tensor')
            if any(p < 0 or p >= tensor.shape[1] for p in positions):
                raise ValueError('Absolute position out of bounds')
            tensor = tensor[0, positions]
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().float().cpu().numpy()
        copy = np.array(tensor, copy=True)
        if not np.isfinite(copy).all():
            raise ValueError('Nonfinite capture')
        self.values[name] = copy

    def finish(self, run_id, required):
        if run_id != self.run_id or self.run_id is None:
            raise ValueError('Stale run')
        if set(required) != set(self.values):
            raise ValueError('Missing or unexpected captures')
        result = self.values
        self.values = {}
        self.run_id = None
        return result
