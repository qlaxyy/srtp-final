"""Identical control tracing in both engineering arms, no sampler mutation."""
class ControlTrace:
    def __init__(self, adapter, max_calls=512):
        if not 1 <= max_calls <= 16000:raise ValueError('Capture budget')
        self.max_calls=max_calls
        self.adapter=adapter;self.runner=adapter.runner
        self.original=self.runner.sampler;self.frames=[]
        self.runner.sampler=self

    def __getattr__(self,name):
        return getattr(self.original,name)

    def __call__(self,logits,batch,**kwargs):
        import torch
        if len(self.frames)>=self.max_calls:raise RuntimeError('Control trace limit')
        a=self.adapter;s=self.runner.steer_vector_state
        idx=batch.idx_mapping[:batch.num_reqs].long().clone()
        pre=torch.stack([s._coefs[idx],s._prev_step_mean[idx],a.opening[idx].float(),
                         a.thinking[idx].float(),a.count[idx].float()],dim=1).clone()
        result=self.original(logits,batch,**kwargs)
        post=torch.stack([a.opening[idx].float(),a.thinking[idx].float(),a.count[idx].float(),
                          a.eligible_count[idx].float(),a.changed_count[idx].float()],dim=1).clone()
        self.frames.append((idx,pre,post))
        return result

    def export(self):
        import numpy as np
        rows={}
        for tensors in self.frames:
            idx,pre,post=[x.detach().cpu().numpy() for x in tensors]
            for i,slot in enumerate(idx):rows.setdefault(int(slot),[]).append(np.concatenate([pre[i],post[i]]))
        return {k:np.stack(v) for k,v in rows.items()}

    def close(self):
        if self.runner.sampler is not self:raise RuntimeError('Close WSC observer first')
        self.runner.sampler=self.original
