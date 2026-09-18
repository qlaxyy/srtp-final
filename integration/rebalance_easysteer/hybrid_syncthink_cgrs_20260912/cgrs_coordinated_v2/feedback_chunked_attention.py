"""Process-local exact query-chunked eager attention; never changes installed code."""
import torch
from torch.utils.checkpoint import checkpoint

def install(chunk=128):
    import transformers.models.qwen2.modeling_qwen2 as q
    original=q.eager_attention_forward
    def forward(module,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
        if dropout != 0: raise ValueError('Only deterministic evaluation supported')
        k=q.repeat_kv(key,module.num_key_value_groups)
        v=q.repeat_kv(value,module.num_key_value_groups)
        def block(x,k,v,mask):
            z=torch.matmul(x,k.transpose(2,3))*scaling
            if mask is not None:z=z+mask
            return torch.matmul(torch.softmax(z,dim=-1,dtype=torch.float32).to(x.dtype),v)
        parts=[]
        for i in range(0,query.shape[2],chunk):
            x=query[:,:,i:i+chunk];mask=None if attention_mask is None else attention_mask[:,:,i:i+chunk,:]
            args=(x,k,v,mask)
            if torch.is_grad_enabled() and any(t.requires_grad for t in [x,k,v]):
                y=checkpoint(block,*args,use_reentrant=False)
            else:y=block(*args)
            parts.append(y)
        return torch.cat(parts,dim=2).transpose(1,2).contiguous(),None
    q.eager_attention_forward=forward
    return lambda:setattr(q,'eager_attention_forward',original)
