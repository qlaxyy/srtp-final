"""CPU PyTorch synthetic hook/slot/buffer tests; not native vLLM validation."""
from types import SimpleNamespace as NS
import json
import numpy as np
import torch
from wsc_shadow import WSCShadow


class Layer(torch.nn.Module):
    def forward(self, positions, hidden, residual):
        return hidden + 0.01, residual + 0.02


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(28)])

    def forward(self, input_ids, positions):
        h = input_ids[:,None].float().repeat(1,1536)
        r = torch.zeros_like(h)
        for layer in self.layers:
            h,r = layer(positions,h,r)
        return h+r


def test():
    WSCShadow(object()).close()  # Disabled mode cannot even inspect runner.
    model = Model()
    def sampler(logits,batch,**kwargs):
        return NS(sampled_token_ids=torch.tensor([[91],[92]]))
    runner = NS(model=NS(model=model),sampler=sampler,vllm_config=NS(
        model_config=NS(enforce_eager=True),compilation_config=NS(mode=0),parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1),
        speculative_config=None,scheduler_config=NS(async_scheduling=False),cache_config=NS(enable_prefix_caching=False)))
    ids=torch.tensor([11,12,13,21,22]);pos=torch.tensor([0,1,2,0,1])
    reference=model(ids,pos).clone()
    obs=WSCShadow(runner,enabled=True,max_calls=2)
    batch=NS(num_draft_tokens=0,num_reqs=2,logits_indices=torch.tensor([2,4]),
        idx_mapping=torch.tensor([7,3]),num_computed_tokens_np=np.array([0,0]),
        num_scheduled_tokens=np.array([3,2]),prefill_len_np=np.array([3,2]),seq_lens=torch.tensor([3,2]))
    assert torch.equal(model(ids,pos),reference)
    logits=torch.zeros(2,100);before=logits.clone()
    assert torch.equal(runner.sampler(logits,batch).sampled_token_ids,torch.tensor([[91],[92]]))
    ids.fill_(999);pos.fill_(999)
    records=obs.export()
    assert records[7][0]['input_id']==13 and records[3][0]['position']==1
    assert records[7][0]['selected']==91 and torch.equal(logits,before)
    assert np.isfinite(records[7][0]['hidden']).all()
    try:runner.sampler(logits,batch)
    except RuntimeError:pass
    else:raise AssertionError('Missing forward accepted')
    obs.close();assert runner.sampler is sampler
    assert not model._forward_pre_hooks and not model.layers[26]._forward_hooks
    assert not model.layers[27]._forward_pre_hooks
    # Request ordering changes: mapping follows batch slots, not row number.
    obs=WSCShadow(runner,enabled=True,max_calls=1)
    model(torch.tensor([91,92]),torch.tensor([3,2]))
    batch.logits_indices=torch.tensor([0,1]);batch.idx_mapping=torch.tensor([3,7]);batch.seq_lens=torch.tensor([4,3])
    runner.sampler(logits,batch);assert obs.export()[3][0]['input_id']==91
    obs.close()
    from score_wsc_shadow import score_capture
    selected=np.array([1]*10+[271]+[1]*10+[271]+[99])
    data=dict(selected=selected,input_ids=np.r_[9,selected[:-1]],
        positions=np.arange(4,4+len(selected)),prompt_tokens=np.array(5),
        hidden=np.ones((len(selected),1536),dtype=np.float32))
    scored=score_capture(data,[271],np.zeros(1536,dtype=np.float32),2.)
    assert [x['chunk_tokens'] for x in scored['events']]==[10,10]
    assert scored['would_trigger'] and not scored['invalid_feature']
    bad=dict(data,input_ids=np.zeros(len(selected),dtype=np.int64))
    try:score_capture(bad,[271],np.zeros(1536,dtype=np.float32),2.)
    except AssertionError:pass
    else:raise AssertionError('Shifted input accepted')
    return dict(default_off_no_access=True,output_and_logits_unchanged=True,
        mapped_slots=True,buffer_copy=True,missing_forward_rejected=True,hooks_removed=True,
        residual_boundary_checked=True,processed_boundary_scoring=True,shift_rejected=True,
        gpu_used=False,native_vllm_validated=False)


if __name__=='__main__':print(json.dumps(test(),indent=2))
