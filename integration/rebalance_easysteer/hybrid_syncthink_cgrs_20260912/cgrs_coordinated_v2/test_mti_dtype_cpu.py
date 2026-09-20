"""A trigger in another request must not change the native lexical rounding."""
import torch
from mti_native_adapter import selected_contrast


def test():
    for dtype in (torch.bfloat16,torch.float32):
        raw=torch.tensor([[1.,3.,-2.],[4.,-1.,2.],[6.,2.,1.]],dtype=dtype)
        original=raw.clone();aux=torch.tensor([[2.,4.,1.]])
        result=selected_contrast(raw,aux,[1])
        assert result.dtype==dtype and torch.equal(raw,original)
        assert torch.equal(result[[0,2]],raw[[0,2]])
        assert torch.equal(result[[0,2]]-.6931471805599453,raw[[0,2]]-.6931471805599453)
        expected=(1.5*raw[1].float()-.5*aux[0]).to(dtype)
        assert torch.equal(result[1],expected)
    # Demonstrates why preserving only numerical row values is insufficient.
    raw=torch.tensor([1.],dtype=torch.bfloat16)
    assert float((raw-.6931471805599453)[0])!=float((raw.float()-.6931471805599453)[0])
    return {'native_dtype_and_unselected_penalty_identity':True,'dtype_confound_reproduced':True,'gpu_used':False}


if __name__=='__main__':print(test())
