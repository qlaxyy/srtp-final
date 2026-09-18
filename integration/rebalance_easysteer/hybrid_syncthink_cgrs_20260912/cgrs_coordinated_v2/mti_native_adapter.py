"""Opt-in native paged branch. Existing R observer and L27 sampler stay intact.

Limited to Qwen2, one full-attention KV group, FA, in-graph additive R, TP1.
Auxiliary cue has no R boundary tokens, so its graph steering masks are zero.
This is our MTI-inspired adaptation, not the upstream runner replacement.
"""
import time
import torch
from mti_reference import entropy


class MTIAdapter:
    def __init__(self, lexical, tokenizer, mode='off'):
        self.mode=mode;self.calls=0;self.rows=0;self.cue_tokens=0;self.host_seconds=0.;self.first={}
        self.lexical=lexical
        if mode=='off':return
        if mode not in ('shadow','active'):raise ValueError(mode)
        self.runner=r=lexical.runner;self.original=r.sampler
        cfg=r.vllm_config
        assert cfg.parallel_config.tensor_parallel_size==1 and cfg.parallel_config.pipeline_parallel_size==1
        assert cfg.steer_vector_config.graph_mode=='in_graph'
        assert cfg.model_config.hf_config.model_type=='qwen2'
        assert not cfg.cache_config.enable_prefix_caching and cfg.speculative_config is None
        groups=r.kv_cache_config.kv_cache_groups
        assert len(groups)==1 and len(r.attn_groups)==1
        spec=groups[0].kv_cache_spec
        self.block_size=spec.block_size
        assert self.block_size>=2 and spec.storage_block_size==self.block_size
        self.layers=list(groups[0].layer_names)
        ctx=cfg.compilation_config.static_forward_context
        self.caches=[]
        for name in self.layers:
            layer=ctx[name]
            assert type(layer.impl).__name__=='FlashAttentionImpl'
            assert layer.impl.sliding_window==(-1,-1)
            cache=layer.kv_cache
            # Pinned FA packs K/V in the last dimension: (blocks, heads, slots, 2*dim).
            assert isinstance(cache,torch.Tensor) and cache.ndim==4
            axis=0
            assert cache.shape[1]==spec.num_kv_heads and cache.shape[2]==self.block_size
            assert cache.shape[3]==2*spec.head_size
            self.caches.append((cache,axis))
        cue=tokenizer.encode('OUTPUT ERROR',add_special_tokens=False)
        assert cue==[30301,12874] and not set(cue)&set(torch.where(lexical.boundary)[0].cpu().tolist())
        self.cue=torch.tensor(cue,device=r.device,dtype=torch.long)
        self.pool=lexical.core.scheduler.kv_cache_manager.block_pool
        self.blocks=self.pool.get_new_blocks(2*r.max_num_reqs)
        self.scratch=torch.tensor([b.block_id for b in self.blocks],device=r.device,dtype=torch.int32).reshape(-1,2)
        self.closed=False;r.sampler=self

    def __getattr__(self,name):
        if name=='original':raise AttributeError(name)
        return getattr(self.original,name)

    def close(self):
        if self.mode=='off' or self.closed:return
        self.runner.sampler=self.original
        torch.cuda.synchronize()
        self.pool.free_blocks(self.blocks);self.closed=True

    def report(self):
        return dict(mode=self.mode,extra_model_forward_count=self.calls,selected_rows=self.rows,
            auxiliary_input_tokens=self.cue_tokens,branch_host_seconds=self.host_seconds,
            first_intervention_counts=self.first,
            timing_note='Branch host interval includes gate synchronization and eager dispatch; not isolated CUDA kernel time')

    def branch(self,batch,selected,lengths):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        from vllm.forward_context import set_forward_context
        from vllm.config import CUDAGraphMode
        r=self.runner;n=len(selected);device=r.device;bs=self.block_size
        sel=torch.tensor(selected,device=device,dtype=torch.long)
        prefix=torch.tensor(lengths,device=device,dtype=torch.long)
        tables,_=r.prepare_attn(batch)
        table=tables[0].index_select(0,sel).clone()
        slots=batch.idx_mapping.index_select(0,sel).long()
        dest=self.scratch[slots]
        full=prefix//bs;tail=prefix%bs
        row=torch.arange(n,device=device)
        assert int(full.max().item())+1<table.shape[1]
        source=table[row,full].long()
        # Copy tail only for non-aligned prefixes; never read a nonexisting next block.
        mask=tail>0
        for cache,axis in self.caches:
            cache.index_copy_(axis,dest[mask,0].long(),cache.index_select(axis,source[mask]))
        table[row,full]=dest[:,0]
        overflow=tail+2>bs
        table[row[overflow],full[overflow]+1]=dest[overflow,1]
        positions=(prefix[:,None]+torch.arange(2,device=device)).reshape(-1)
        cue_slots=(table[row[:,None],positions.reshape(n,2)//bs].long()*bs+positions.reshape(n,2)%bs).reshape(-1)
        q=torch.arange(0,2*n+1,2,device=device,dtype=torch.int32)
        meta=FlashAttentionMetadata(num_actual_tokens=2*n,max_query_len=2,query_start_loc=q,
            max_seq_len=max(lengths)+2,seq_lens=(prefix+2).int(),block_table=table,
            slot_mapping=cue_slots,use_cascade=False,common_prefix_len=0,
            cu_prefix_query_lens=None,prefix_kv_lens=None,suffix_kv_lens=None)
        managers=r.steer_vector_manager.graph_batch_entries()
        controllers={id(m):m for _,_,ms in managers.values() for m in ms}
        masks=[(m.graph_mask,m.graph_mask.clone()) for m in controllers.values()]
        try:
            for target,_ in masks:target.zero_()
            with set_forward_context({name:meta for name in self.layers},r.vllm_config,
                    num_tokens=2*n,cudagraph_runtime_mode=CUDAGraphMode.NONE,skip_compiled=True,
                    slot_mapping={name:cue_slots for name in self.layers},
                    is_padding=torch.zeros(2*n,device=device,dtype=torch.bool)):
                hidden=r.model(input_ids=self.cue.repeat(n),positions=positions)
            return r.model.compute_logits(hidden[1::2]).float()
        finally:
            for target,saved in masks:target.copy_(saved)

    def __call__(self,logits,batch,**kwargs):
        began=time.monotonic();o=self.lexical
        idx=batch.idx_mapping[:batch.num_reqs].long()
        valid=torch.as_tensor(~batch.is_prefilling_np[:batch.num_reqs],device=logits.device)
        selected=torch.where(valid & o.thinking[idx] & (entropy(logits)>.5))[0].cpu().tolist()
        if selected:
            lengths=batch.seq_lens[selected].cpu().tolist()
            aux=self.branch(batch,selected,lengths)
            self.calls+=1;self.rows+=len(selected);self.cue_tokens+=2*len(selected)
            counts=o.count[idx[selected]].cpu().tolist()
            for row,count in zip(selected,counts):self.first.setdefault(batch.req_ids[row],count)
            if self.mode=='active':
                logits=logits.float().clone()
                logits[selected]=1.5*logits[selected]-.5*aux
        self.host_seconds+=time.monotonic()-began
        return self.original(logits,batch,**kwargs)
