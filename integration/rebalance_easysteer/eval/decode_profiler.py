"""Opt-in, bounded CPU/CUDA tracing; never enable for reported speed numbers."""
from pathlib import Path
import json


class ProfileWindowComplete(Exception):
    """Requested diagnostic window is saved; do not finish a full evaluation."""


def mark_runtime_ranges(runner):
    """Name existing operations only in diagnostic runs; leave kernels unchanged."""
    import torch
    from functools import wraps
    def wrap(owner, method, label):
        original = getattr(owner, method)
        @wraps(original)
        def marked(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        setattr(owner, method, marked)
    wrap(runner, "sample", "rebalance.logits_sampling_and_control")
    wrap(runner.steer_vector_state, "observe_sample", "rebalance.dynamic_state")


class EngineStepProfiler:
    def __init__(self, path, start_step=16, steps=12, factory=None, stop_after_window=False):
        if start_step < 0 or steps <= 0:
            raise ValueError("Invalid profiling window")
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.start_step, self.steps = start_step, steps
        self.factory = factory
        self.index = 0
        self.profiler = None
        self.finished = False
        self.stop_after_window = stop_after_window

    def run_step(self, callback):
        if self.index == self.start_step:
            if self.factory is None:
                import torch
                def factory():
                    return torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ])
                self.factory = factory
            self.profiler = self.factory()
            self.profiler.start()
        result = callback()
        self.index += 1
        if self.profiler is not None and not self.finished:
            self.profiler.step()
            if self.index >= self.start_step + self.steps:
                self.close()
                if self.stop_after_window:
                    raise ProfileWindowComplete(str(self.path))
        return result

    def close(self):
        if self.profiler is not None and not self.finished:
            self.finished = True
            self.profiler.stop()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.profiler.export_chrome_trace(str(self.path))
            if hasattr(self.profiler, "key_averages"):
                rows = [dict(name=e.key, calls=e.count, cpu_us=e.cpu_time_total,
                             self_cpu_us=e.self_cpu_time_total,
                             device_us=e.device_time_total,
                             self_device_us=e.self_device_time_total)
                        for e in self.profiler.key_averages()]
                self.path.with_suffix(".operators.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8")
