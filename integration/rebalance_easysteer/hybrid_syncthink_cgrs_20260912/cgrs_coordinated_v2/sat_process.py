"""SAT-code-compatible features and a proposed RC14 protection state machine.

No generation hooks, downloads, or active intervention on import. This is our
adaptation of the pinned SAT contract, not the official SAT generation method.
Scores concern prefix quality, not the causal usefulness of further reflection.
"""
from collections import deque
import math
import numpy as np

FEATURES = ('canonical_entropy', 'canonical_margin', 'canonical_z_logp',
            'canonical_selected_rank', 'canonical_logprobs', 'canonical_logit_gap',
            'canonical_topk_mass@5', 'canonical_topk_mass@10',
            'canonical_d_entropy', 'canonical_d_margin', 'canonical_d_logp')


class ProcessFeatures:
    """One instance per request. Invalid evidence latches fallback to RC14.

    Consume accepted thinking tokens, including newline tokens. Empty lines do
    not close a step. A mixed content/newline token closes one step, matching
    the public runner's token-granular segmentation. Never score an unfinished
    tail, prompt tokens, or the answer outside thinking. Delta/z windows persist
    across steps. A new request needs a new instance, not a recycled GRU state.
    """
    def __init__(self, stats):
        if set(stats) != set(FEATURES):
            raise ValueError('Exact frozen 11-feature statistics required')
        self.stats = np.asarray([stats[k] for k in FEATURES], dtype=np.float64)
        if self.stats.shape != (11, 2) or not np.isfinite(self.stats).all() or (self.stats[:, 1] <= 1e-12).any():
            raise ValueError('Invalid statistics')
        self.logp_window = deque(maxlen=50)
        self.previous = None
        self.invalid_reason = None
        self.rows, self.pieces = [], []
        self.tokens = self.start = 0
        self.has_content = False

    def fail(self, reason):
        self.invalid_reason = reason
        self.rows.clear()
        self.pieces.clear()
        return None

    def accept(self, values, ids, selected, piece):
        if self.invalid_reason:
            return None
        z, ids = np.asarray(values, dtype=np.float64), np.asarray(ids)
        if z.shape != (512,) or ids.shape != (512,) or not np.isfinite(z).all():
            return self.fail('invalid_top512')
        if (np.diff(z) > 0).any() or len(set(ids.tolist())) != 512:
            return self.fail('unsorted_or_duplicate_top512')
        where = np.flatnonzero(ids == selected)
        if len(where) != 1:
            return self.fail('accepted_token_outside_top512')
        logp = z - z.max() - np.log(np.exp(z-z.max()).sum())
        p = np.exp(logp)
        entropy, margin, chosen = float(-(p*logp).sum()), float(p[0]-p[1]), float(logp[where[0]])
        self.logp_window.append(chosen)
        w = np.asarray(self.logp_window)
        delta = np.zeros(3) if self.previous is None else np.array([entropy, margin, chosen])-self.previous
        row = np.array([entropy, margin, (chosen-w.mean())/(w.std()+1e-8),
                        float(where[0]+1), chosen, z[0]-z[1], p[:5].sum(), p[:10].sum(), *delta])
        self.previous = np.array([entropy, margin, chosen])
        self.rows.append(row)
        self.pieces.append(piece)
        self.tokens += 1
        normal = piece.replace('\r\n', '\n')
        self.has_content |= bool(normal.replace('\n', '').strip())
        if '\n' not in normal or not self.has_content:
            return None
        matrix = np.asarray(self.rows, dtype=np.float32)
        matrix[:, 3] = np.log1p(np.maximum(0, matrix[:, 3]))
        matrix = ((matrix-self.stats[:, 0])/self.stats[:, 1]).astype(np.float32)
        vector = np.stack([matrix.mean(0), matrix.max(0), matrix[-1]], axis=1).reshape(33)
        result = dict(text=''.join(self.pieces), features=vector, start=self.start, end=self.tokens)
        self.rows.clear(); self.pieces.clear()
        self.start = self.tokens
        self.has_content = False
        return result


class ProtectionState:
    """Proposed, unvalidated guard: 5 scores <=.4 enter; >.5 exit.

    This exposes a veto only; it cannot force stopping or alter ReBalance.
    An invalid score disables the guard for this request rather than inventing
    evidence. Thresholds are design choices, not newly fitted parameters.
    """
    def __init__(self):
        self.low_run = 0
        self.protected = False
        self.invalid = False

    def update(self, score):
        if self.invalid:
            return False
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            self.invalid = True
            self.protected = False
            return False
        self.low_run = self.low_run+1 if score <= .4 else 0
        if score > .5:
            self.protected = False
        elif self.low_run >= 5:
            self.protected = True
        return self.protected


def make_gru(checkpoint, device='cpu'):
    """Load only tensors from the pinned public checkpoint; return eval model."""
    import hashlib
    from pathlib import Path
    import torch
    from torch import nn
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != '628c182133004afee4aa07077afc899d0abbf126b3630c99585d17863b29f629':
        raise ValueError('Unpinned SAT checkpoint')

    class StepGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(417, 128)
            self.gru = nn.GRU(128, 128, batch_first=True)
            self.out_head = nn.Linear(128, 1)

        def forward(self, features, embeddings, hidden=None):
            projected = self.input_proj(torch.cat([features, embeddings], dim=-1))
            y, hidden = self.gru(projected, hidden)
            return self.out_head(y).squeeze(-1), hidden

    model = StepGRU()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    return model.eval().to(device)
