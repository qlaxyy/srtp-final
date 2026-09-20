"""Existing inference Python, native Torch CPU only, no model initialization."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import check_runtime as old


def main():
    spec=importlib.util.spec_from_file_location('previous_soft_checks',HERE.parent/'soft2_20260913/check_torch.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    previous.main()
    for dtype in (torch.float32,torch.bfloat16):
        cfg=dict(mode='mix05',entropy_weight=.8,pacing_cap=64,end_token_id=151649)
        s=old.module.HybridTerminationState(3,'cpu')
        s.add_request('mix',2,[151648],1,cfg)
        s.add_request('shadow',0,[151648],1,dict(cfg,mode='shadow'))
        s.count[[2,0]]=63  # Synthetic accepted prefix, after the pacing ramp.
        raw=torch.full((2,151650),-40.,dtype=dtype);raw[:,7]=10;raw[:,151649]=0
        before=raw.clone();ticket=s.read_apply(raw,old.batch(['mix','shadow'],[2,0],[64,64]))
        assert torch.equal(raw,before)
        x=torch.full_like(raw,-torch.inf);x[:,7]=0;before=x.clone()
        s.apply_after_filter(x)
        q=torch.softmax(x.float(),dim=-1)
        assert abs(float(q[0,151649])-.05)<.0005 and q[1,151649]==0
        assert torch.equal(x[:,:151649],before[:,:151649])
        assert int(s.revived_end_count[2])==1 and int(s.bias_count[2])==1
        twice=x.clone();s.apply_after_filter(x);assert torch.equal(twice,x)
        s.observe(ticket,torch.tensor([[7],[7]]))
        # Reorder, close on the actual sampled end, then reuse the slot.
        ticket=s.read_apply(raw,old.batch(['shadow','mix'],[0,2],[65,65]))
        x=before.clone();s.apply_after_filter(x)
        assert torch.isfinite(x[1,151649]) and torch.isneginf(x[0,151649])
        s.observe(ticket,torch.tensor([[7],[151649]]))
        ticket=s.read_apply(raw,old.batch(['shadow','mix'],[0,2],[66,66]))
        x=before.clone();s.apply_after_filter(x);assert torch.equal(x,before)
        s.remove_request('mix');s.add_request('reused',2,[151648],1,dict(cfg,mode='soft'))
        assert not s.mix_slots[2] and s.revived_end_count[2]==0 and s.first_bias[2]==-1
        # Distribution identity for previously retained end and singleton end.
        v=torch.tensor([[0.,-1.,-2.],[-torch.inf,-torch.inf,0.]],dtype=dtype)
        p=torch.softmax(v.float(),-1);y=v.clone();y[:,2]=old.module.mixed_end_column(v,2).to(dtype)
        expected=.95*p;expected[:,2]+=.05
        assert torch.allclose(torch.softmax(y.float(),-1),expected,atol=.0005)
        print(json.dumps(dict(status='pass',dtype=str(dtype),device='cpu',forwards=0,
                              cases='mixture law; revive excluded end; shadow; reorder; close; ticket; reset; retained end')))


if __name__=='__main__':main()
