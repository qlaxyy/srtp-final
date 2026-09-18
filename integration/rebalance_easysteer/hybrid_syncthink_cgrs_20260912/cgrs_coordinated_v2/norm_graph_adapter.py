"""Process-local full-graph residual-aware correction, no shared source edits."""
import torch


def make_kernel(original, enabled, counters):
    def kernel(tables, graph_mask, replace_mask, normalize_flag, token_rows,
               hidden_states, residual):
        old = original(tables, graph_mask, replace_mask, normalize_flag,
                       token_rows, hidden_states, residual)
        x = hidden_states.float()
        y = old.float()
        if residual is not None:
            x = x + residual.float()
            y = y + residual.float()
        nx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        ny = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        valid = (nx > 0) & (ny > 0) & torch.isfinite(nx) & torch.isfinite(ny)
        n = hidden_states.shape[0]
        active = ((graph_mask[:n] != 0) & (token_rows[:n] > 0) & (enabled[0] != 0)).unsqueeze(1)
        ratio = nx / torch.where(valid, ny, torch.ones_like(ny))
        corrected = (old.float() + y * ratio - y).to(old.dtype)
        candidate = torch.where(valid, corrected, hidden_states)
        counters.add_(torch.stack((active.sum(), (active & ~valid).sum())).to(counters.dtype))
        return torch.where(active, candidate, old)
    return kernel


class NormGraphAdapter:
    def __init__(self):
        from vllm.steer_vectors import controllers
        self.controllers = controllers
        self.original = controllers.apply_decoder_families
        self.enabled = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.counters = torch.zeros(2, device='cuda', dtype=torch.int64)
        controllers.apply_decoder_families = make_kernel(self.original,self.enabled,self.counters)

    def set_mode(self, enabled):
        torch.cuda.synchronize()
        self.enabled.fill_(int(enabled))
        self.counters.zero_()

    def report(self):
        torch.cuda.synchronize()
        active, fallback = self.counters.tolist()
        return dict(enabled=bool(self.enabled.item()),active_rows=active,fallback_rows=fallback)

    def close(self):
        self.controllers.apply_decoder_families = self.original
