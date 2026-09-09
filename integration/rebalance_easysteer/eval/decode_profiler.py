"""Opt-in, bounded CPU/CUDA tracing; never enable for reported speed numbers."""
from pathlib import Path


class EngineStepProfiler:
    def __init__(self, path, start_step=16, steps=12, factory=None):
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
        return result

    def close(self):
        if self.profiler is not None and not self.finished:
            self.finished = True
            self.profiler.stop()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.profiler.export_chrome_trace(str(self.path))
