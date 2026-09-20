"""Exercise the actual sampler on CPU Torch, including BF16 and row routing."""
import copy,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as N
import torch
from adapter import Adapter,Sampler
from policy import TRIGGERS,PENALTY
from test_native import fixture

def phrase(o):
    o.lexical_mode='but_wait';o.after_but=torch.zeros_like(o.opening)
    o.but_lookup=torch.zeros_like(o.clean);o.but_lookup[[3983,1988,8088,714]]=True
    o.but_columns=torch.tensor([i in (3983,1988,8088,714) for i in TRIGGERS])
    o.wait_columns=torch.tensor([i in (14190,13824,11489,3783) for i in TRIGGERS])
    o.phrase_count=torch.zeros_like(o.count)

def tick(o,b,tokens,dtype):
    b.seq_lens=o.prompt_len[b.idx_mapping]+o.count[b.idx_mapping]
    o.original_sampler=lambda *args,**kw:N(sampled_token_ids=torch.tensor(tokens)[:,None])
    x=torch.ones(4,151936,dtype=dtype);Sampler(o)(x,b);return x

class Tests(unittest.TestCase):
    def test_causal_phrase_whitespace_and_content_reset(self):
        for dtype in (torch.float32,torch.bfloat16):
            o,b,_=fixture();phrase(o)
            x=tick(o,b,[-1,3983,271,220],dtype)
            self.assertEqual(float(x[1,3983]),1.)
            self.assertLess(float(x[1,14190]),1.)
            self.assertTrue(o.after_but[0]);self.assertFalse(o.opening[0])
            x=tick(o,b,[-1,220,271,220],dtype)
            self.assertLess(float(x[1,3783]),1.);self.assertEqual(float(x[1,3983]),1.)
            self.assertTrue(o.after_but[0])
            tick(o,b,[-1,1000,271,220],dtype)
            self.assertFalse(o.after_but[0])
            self.assertTrue(torch.all(tick(o,b,[-1,3783,271,220],dtype)[1]==1))

    def test_answer_boundary_and_nonnegative_reset(self):
        for token in (151649,151648,271):
            o,b,_=fixture();phrase(o);tick(o,b,[-1,3983,271,220],torch.float32)
            tick(o,b,[-1,token,271,220],torch.float32);self.assertFalse(o.after_but[0])
        for coefficient in (0.,.1,float('nan')):
            o,b,_=fixture();phrase(o);o.after_but[0]=True;o.opening[0]=False
            o.runner.steer_vector_state._coefs[0]=coefficient
            self.assertTrue(torch.all(tick(o,b,[-1,3783,271,220],torch.float32)[1]==1))

    def test_default_same_as_committed_sampler(self):
        root=Path(__file__).resolve().parents[4];rel=Path(__file__).with_name('adapter.py').relative_to(root).as_posix()
        scope={};exec(subprocess.check_output(['git','-C',str(root),'show','785ac68:'+rel],text=True),scope)
        for dtype in (torch.float32,torch.bfloat16):
            a,b,x=fixture(dtype=dtype);c,d,y=fixture(dtype=dtype)
            Sampler(a)(x,b);scope['Sampler'](c)(y,d)
            self.assertTrue(torch.equal(x,y))
            for k in ('opening','thinking','count','changed_count','first_change'):
                self.assertTrue(torch.equal(getattr(a,k),getattr(c,k)))

    def test_off_has_no_access(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError(name)
        Adapter(Trap(),Trap(),lexical_mode='but_wait').close()

if __name__=='__main__':unittest.main(verbosity=2)
