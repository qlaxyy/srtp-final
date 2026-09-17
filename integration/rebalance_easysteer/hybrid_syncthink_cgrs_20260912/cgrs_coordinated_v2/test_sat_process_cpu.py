"""CPU contracts; synthetic evidence, never a compression/accuracy result."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Dict, List, Optional, Tuple
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat_process import FEATURES, ProcessFeatures, ProtectionState, make_gru
from sat_shadow import ShadowRecorder
from test_native import fixture
from adapter import Sampler

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'.codex_work/literature_broadening_20260917/sat'
ASSETS = ROOT/'.codex_work/sat_compatibility_20260917'


class Contracts(unittest.TestCase):
    def test_feature_and_checkpoint_parity(self):
        import hashlib
        raw=(SOURCE/'run_think_control_V5_ds_qwen.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '1f5d345cedd67226380b3c8217d4fc9c79cf73c0a5dfa5d8fe9c74f3b281b419')
        nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
               and n.name in ('update_canonical_feats_incremental','StepSeqPRM_GRU')]
        scope=dict(globals()); exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_sat','exec'),scope)
        stats=json.loads((SOURCE/'zstats.json').read_text())
        p=ProcessFeatures(stats); observed={k:[] for k in FEATURES}
        rng=np.random.default_rng(42); steps=[]
        for i in range(80):
            vals=np.sort(rng.normal(size=512).astype('float32'))[::-1].copy(); ids=np.arange(512)
            selected=i%23
            scope['update_canonical_feats_incremental'](observed,dict(topk_vals=torch.tensor(vals[None]),topk_idx=torch.tensor(ids[None])),selected,50)
            step=p.accept(vals,ids,selected,'word\n' if i in (10,39,79) else 'word')
            if step is not None:
                matrix=[]
                for key in FEATURES:
                    x=np.asarray(observed[key][step['start']:step['end']],dtype='float32')
                    if key=='canonical_selected_rank': x=np.log1p(x)
                    m,s=stats[key];matrix.append((x-m)/s)
                matrix=np.stack(matrix,axis=1)
                expected=np.stack([matrix.mean(0),matrix.max(0),matrix[-1]],axis=1).reshape(33)
                np.testing.assert_allclose(step['features'],expected,atol=5e-5,rtol=0)
                steps.append(step['features'])
        self.assertEqual(len(steps),3)
        weights=ASSETS/'step_seq_prm_gte_small_logits_gru_best_psr2.pt'
        ours=make_gru(weights); official=scope['StepSeqPRM_GRU'](33,384).eval()
        official.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'),strict=True)
        x=torch.tensor(np.array(steps))[None]; emb=F.normalize(torch.tensor(rng.normal(size=(1,3,384)),dtype=torch.float32),dim=-1)
        with torch.no_grad():
            expected=official(x,emb,torch.tensor([3])); got,h=ours(x,emb)
            h=None; parts=[]
            for i in range(3):
                y,h=ours(x[:,i:i+1],emb[:,i:i+1],h);parts.append(y)
        torch.testing.assert_close(got,expected,atol=1e-5,rtol=0)
        torch.testing.assert_close(torch.cat(parts,1),expected,atol=1e-5,rtol=0)

    def test_invalid_latches_and_new_request_resets(self):
        stats={k:(0,1) for k in FEATURES}; vals=np.arange(512)[::-1];ids=np.arange(512)
        p=ProcessFeatures(stats)
        self.assertIsNone(p.accept(vals,ids,999,'bad\n'))
        self.assertIsNone(p.accept(vals,ids,0,'valid\n'))
        self.assertEqual(p.invalid_reason,'accepted_token_outside_top512')
        q=ProcessFeatures(stats)
        self.assertIsNone(q.accept(vals,ids,0,'\n\n'))
        step=q.accept(vals,ids,0,'text\nmore')
        self.assertEqual((step['start'],step['end'],step['text']),(0,2,'\n\ntext\nmore'))
        self.assertEqual(q.tokens,2)
        self.assertIsNone(q.accept(vals,ids,0,'unfinished'))

    def test_protection_hysteresis_invalid_and_reset(self):
        p=ProtectionState()
        self.assertEqual([p.update(.4) for _ in range(5)],[False]*4+[True])
        self.assertTrue(p.update(.5));self.assertFalse(p.update(.501))
        self.assertFalse(p.update(float('nan')))
        self.assertFalse(any(p.update(.1) for _ in range(6)))
        self.assertFalse(ProtectionState().protected)

    def test_shadow_matches_RC14_bfloat16_and_partial_prefill(self):
        # Record pre-penalty values even though the inner adapter mutates logits.
        for dtype in (torch.float32,torch.bfloat16):
            a,b,x=fixture(dtype=dtype); c,d,y=fixture(dtype=dtype)
            x[:,a.ids]=3; y.copy_(x)
            runner=N(sampler=Sampler(c)); original=runner.sampler
            recorder=ShadowRecorder(runner,enabled=True,max_calls=1)
            ra=Sampler(a)(x,b); rc=runner.sampler(y,d)
            self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(ra.sampled_token_ids,rc.sampled_token_ids))
            for field in ('count','eligible_count','changed_count','first_change','opening','thinking'):
                self.assertTrue(torch.equal(getattr(a,field),getattr(c,field)),field)
            rows=recorder.export()
            self.assertEqual(set(rows),{0,1,2});self.assertEqual(rows[0][0]['selected'],151649)
            self.assertEqual(float(rows[0][0]['values'][0]),3.)
            self.assertEqual(float(y[1,a.ids[0]]),float(x[1,a.ids[0]]))
            with self.assertRaises(RuntimeError):runner.sampler(y,d)
            recorder.close();self.assertIs(runner.sampler,original)

    def test_default_off_has_no_hooks_or_buffers(self):
        runner=N(sampler=object());before=runner.sampler
        recorder=ShadowRecorder(runner);recorder.close()
        self.assertIs(runner.sampler,before)
        self.assertEqual(recorder.__dict__,{'enabled':False})


if __name__=='__main__':unittest.main(verbosity=2)
