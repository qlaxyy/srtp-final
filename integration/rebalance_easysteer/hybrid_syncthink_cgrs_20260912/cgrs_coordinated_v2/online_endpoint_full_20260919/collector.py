"""Bounded eager capture, exported in chunks; only step starts retain hidden data."""
from observer import EndpointObserver,NativeAudit
import numpy as np

class SparseRows:
    def __init__(self,boundaries):
        self.boundaries=set(boundaries);self.last={};self.metadata={};self.hidden={}
    def add(self,slots,positions,ids,selected,pre,post):
        for i,slot in enumerate(slots):
            slot=int(slot);pos=int(positions[i]);token=int(ids[i])
            previous=self.last.get(slot)
            if previous is not None:
                assert pos==previous[0]+1 and token==previous[2],'Noncontiguous accepted prefix'
            # The first captured input is the last prompt token. The second is
            # generated token zero. A boundary input itself is never a step.
            count=len(self.metadata.get(slot,[]))
            keep=(count==1 or (previous is not None and previous[1] in self.boundaries))
            keep=keep and token not in self.boundaries and token!=151649
            if keep:self.hidden.setdefault(slot,[]).append((pos,pre[i].copy(),post[i].copy()))
            self.metadata.setdefault(slot,[]).append((pos,token,int(selected[i])))
            self.last[slot]=(pos,token,int(selected[i]))

class ChunkObserver(EndpointObserver):
    def __init__(self,runner,boundaries,chunk_size=128):
        super().__init__(runner,enabled=True,max_calls=16000)
        self.chunk_size=chunk_size;self.total_calls=0;self.rows=SparseRows(boundaries)
    def __call__(self,logits,batch,**kwargs):
        t=self.torch;f=self.pending;self.total_calls+=1
        if self.total_calls>16000 or f is None or not f.get('checked'):raise RuntimeError('Capture contract')
        if batch.num_draft_tokens or batch.num_reqs>64:raise RuntimeError('Batch contract')
        take=batch.logits_indices.long();idx=batch.idx_mapping[:batch.num_reqs].long().clone()
        valid=t.as_tensor(batch.num_computed_tokens_np+batch.num_scheduled_tokens>=batch.prefill_len_np,device=logits.device)
        t._assert_async(valid.all(),'Partial prefill unsupported')
        pos,ids,pre,post=[f[k][take].clone() for k in ['positions','ids','pre','hidden']]
        t._assert_async((pos==batch.seq_lens[:batch.num_reqs]-1).all(),'Position mismatch')
        result=self.original(logits,batch,**kwargs)
        if result.sampled_token_ids.shape!=(batch.num_reqs,1):raise RuntimeError('Sample shape')
        self.frames.append((idx,pos,ids,result.sampled_token_ids[:,0].clone(),pre,post));self.pending=None
        if len(self.frames)>=self.chunk_size:self.flush()
        return result
    def flush(self):
        if not self.frames:return
        arrays=[self.torch.cat([f[i] for f in self.frames]).detach() for i in range(6)]
        arrays=[x.float().cpu().numpy() if x.is_floating_point() else x.cpu().numpy() for x in arrays]
        self.rows.add(*arrays);self.frames.clear()
    def export(self):self.flush();return self.rows

class ChunkAudit(NativeAudit):
    def __init__(self,runner):super().__init__(runner,128);self.saved={};self.total_calls=0
    def record(self,batch,tokens,probs):
        self.total_calls+=1
        if self.total_calls>16000:raise RuntimeError('Native clock overflow')
        result=super().record(batch,tokens,probs)
        if len(self.frames)==128:self.flush()
        return result
    def flush(self):
        if not self.frames:return
        import torch
        a=[torch.cat([f[i] for f in self.frames]).detach().cpu().numpy() for i in range(6)]
        for slot in np.unique(a[0]):
            mask=a[0]==slot
            data=np.column_stack([a[1][mask,0],*[x[mask] for x in a[2:]]]).astype(np.float64)
            self.saved.setdefault(int(slot),[]).append(data)
        self.frames.clear()
    def export(self):self.flush();return {k:np.concatenate(v) for k,v in self.saved.items()}
