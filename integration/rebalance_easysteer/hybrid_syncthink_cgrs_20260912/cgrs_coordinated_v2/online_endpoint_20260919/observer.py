"""Eager engineering observer; native intervention/KV history is untouched."""
from wsc_shadow import WSCShadow

class EndpointObserver(WSCShadow):
    def __init__(self,runner,enabled=False,max_calls=512):
        super().__init__(runner,enabled=enabled,max_calls=max_calls)
        if not enabled:return
        # Reuse established input/accepted-token alignment, change measured layer.
        for handle in self.handles[1:]:handle.remove()
        self.handles=self.handles[:1]
        self.pre_frames=[]
        self.handles.append(self.model.layers[20].register_forward_hook(self.before_steering,prepend=True))
        self.handles.append(self.model.layers[20].register_forward_hook(self.after_layer))
        self.handles.append(self.model.layers[21].register_forward_pre_hook(self.before_last))

    def before_steering(self,module,args,output):
        hidden,residual=output
        if self.pending is None or residual is None:raise RuntimeError('Missing residual stream')
        self.pending['pre']=(hidden+residual).clone()

    def __call__(self,logits,batch,**kwargs):
        pre=self.pending['pre'][batch.logits_indices.long()].clone()
        result=super().__call__(logits,batch,**kwargs)
        self.pre_frames.append(pre)
        return result

    def export(self):
        rows=super().export();offset={}
        for frame,pre in zip(self.frames,self.pre_frames):
            idx,valid=frame[:2]
            for i,(slot,ok) in enumerate(zip(idx.cpu().tolist(),valid.cpu().tolist())):
                if ok:
                    n=offset.get(slot,0);rows[slot][n]['pre_hidden']=pre[i].float().cpu().numpy();offset[slot]=n+1
        return rows

class NativeAudit:
    """Records the exact pmax argument and controller state after observe_sample."""
    def __init__(self,runner,max_calls=512):
        self.state=runner.steer_vector_state;self.original=self.state.observe_sample
        self.frames=[];self.max_calls=max_calls;self.state.observe_sample=self.record
    def record(self,batch,tokens,probs):
        if len(self.frames)>=self.max_calls:raise RuntimeError('Native audit overflow')
        idx=batch.idx_mapping[:batch.num_reqs].long().clone()
        # Synchronous, unchunked prefill is required by EndpointObserver/runner.
        if batch.num_draft_tokens:raise RuntimeError('Speculative tokens unsupported')
        sample=tokens.clone();p=probs.clone()
        result=self.original(batch,tokens,probs)
        s=self.state
        self.frames.append((idx,sample,p,s._coefs[idx].clone(),s._prev_step_mean[idx].clone(),s._step_tok_count[idx].clone()))
        return result
    def export(self):
        import numpy as np
        rows={}
        for frame in self.frames:
            arrays=[x.detach().cpu().numpy() for x in frame]
            for i,slot in enumerate(arrays[0]):
                rows.setdefault(int(slot),[]).append(np.array([arrays[1][i,0],*[x[i] for x in arrays[2:]]],dtype=np.float64))
        return {k:np.stack(v) for k,v in rows.items()}
    def close(self):self.state.observe_sample=self.original
