"""Read pinned public WSC tensors without unrestricted pickle execution.

Shape/numerics cannot establish the missing trained feature-layer identity.
Does not import upstream prober/training modules, load LLMs, or contact a server.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import numpy as np
import torch

PINS={'1.5B':('a2c0892a807f7bf1c4d8d954fe2262f0c87b8a91e45da7c260711bba6e501feb',1536),
      '7B':('647a15a0bab8a3e67f6a796c401021d69b4e57fa61ad226fbb2ca557d2d3fee0',3584)}


def storage_from_bytes(data):
    if len(data)>1024*1024:raise ValueError('Unexpected nested tensor size')
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)


class TensorOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):OrderedDict,
                 ('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,
                 ('torch.storage','_load_from_bytes'):storage_from_bytes}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Unapproved global: {module}.{name}')
        return allowed[module,name]


def audit(folder):
    torch.set_num_threads(1);results={}
    for name,(digest,dim) in PINS.items():
        path=folder/(name+'.pkl');raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        obj=TensorOnlyUnpickler(io.BytesIO(raw)).load()
        assert set(obj)=={'model_state_dict','cfg'}
        state,cfg=obj['model_state_dict'],obj['cfg']
        assert set(state)=={'weight','bias'} and cfg['in_dim']==dim
        assert state['weight'].shape==(1,dim) and state['bias'].shape==(1,)
        assert all(x.device.type=='cpu' and torch.isfinite(x).all() for x in state.values())
        model=torch.nn.Linear(dim,1);model.load_state_dict(state,strict=True);model.eval()
        rng=np.random.default_rng(42);x=rng.normal(size=(16,dim)).astype(np.float32)
        with torch.inference_mode():y=model(torch.tensor(x)).numpy()
        reference=x.astype(np.float64)@state['weight'].numpy().astype(np.float64).T+state['bias'].numpy().astype(np.float64)
        err=float(np.max(np.abs(y-reference)));assert err<1e-4
        results[name]=dict(sha256=digest,bytes=len(raw),config=cfg,
            shapes={k:list(v.shape) for k,v in state.items()},dtype=str(state['weight'].dtype),
            weight_norm=float(state['weight'].norm()),bias=float(state['bias'][0]),
            numpy_float64_vs_torch_float32_max_abs_error=err,
            feature_layer_in_checkpoint='absent',feature_token_position_in_checkpoint='absent',
            trained_source_commit_in_checkpoint='absent')
    # Prove arbitrary pickle globals cannot be invoked through this reader.
    for unsafe in (b'cbuiltins\neval\n.', b'cos\nsystem\n.'):
        try:TensorOnlyUnpickler(io.BytesIO(unsafe)).load()
        except pickle.UnpicklingError:pass
        else:raise AssertionError('Unsafe global accepted')
    return dict(results=results,restricted_global_rejection=True,gpu_used=False,
                llm_forward_calls=0,semantic_accuracy_not_evaluated=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.folder)
    with a.output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,indent=2))
