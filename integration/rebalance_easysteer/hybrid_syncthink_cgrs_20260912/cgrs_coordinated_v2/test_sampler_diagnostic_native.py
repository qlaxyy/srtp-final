"""Real Torch CPU contract; execute in existing server environment before CUDA."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as N
import torch
from test_native import fixture
from sampler_diagnostic import FIELDS,PATHS,check_case,SamplerDiagnostic


def setup(dtype):
    o,b,x=fixture(mode='shadow',dtype=dtype)
    o.history_gate='after_first_reflection';o.first_reflection=torch.full_like(o.count,-1)
    o.reflection_lookup=torch.zeros_like(o.clean);o.reflection_lookup[o.ids]=True
    return o,b,x


class NativeDiagnosticTests(unittest.TestCase):
    def test_case_replays_preserve_live_state_with_BF16_and_permuted_partial_prefill(self):
        for dtype in (torch.float32,torch.bfloat16):
            o,b,x=setup(dtype);snapshot={k:getattr(o,k).clone() for k in FIELDS}
            def native(logits,batch):return N(sampled_token_ids=logits.argmax(-1).reshape(-1,1))
            baseline=native(x,b).sampled_token_ids.clone()
            for kind in PATHS:
                result=check_case(o,native,x,b,kind,baseline)
                self.assertTrue(result['passed'],result)
            for k,v in snapshot.items():self.assertTrue(torch.equal(v,getattr(o,k)),k)
            wrong=baseline+1
            self.assertFalse(check_case(o,native,x,b,'native_repeat',wrong)['passed'])

    def test_probe_persists_capture_and_reports_failure_instead_of_advancing(self):
        o,b,x=setup(torch.bfloat16)
        class Native:
            sampling_states=N(seeds=N(gpu=torch.full((4,),42)))
            def __call__(self,logits,batch):
                return N(sampled_token_ids=logits.argmax(-1).reshape(-1,1),
                         num_sampled=torch.ones(4,dtype=torch.int32),num_rejected=torch.zeros(4,dtype=torch.int32))
        o.original_sampler=Native();o.active={f'q{i}':i for i in range(4)}
        o.runner.req_states=N(req_id_to_index=o.active)
        o.runner.steer_vector_state._dynamic_indices=dict(o.active)
        b.positions=torch.tensor([1,29,49,39]);b.logits_indices=torch.arange(4)
        targets={f'q{i}':{int(o.count[i])} for i in range(4)}
        with tempfile.TemporaryDirectory() as tmp:
            probe=SamplerDiagnostic(o,targets,Path(tmp),16*1024*1024)
            result=probe(x,b)
            self.assertEqual(probe.checks,1);self.assertEqual(len(probe.seen),3)
            self.assertTrue((Path(tmp)/'capture_000.pt').exists())
            self.assertTrue((Path(tmp)/'checks.jsonl').exists())
            self.assertEqual(result.sampled_token_ids.shape,(4,1))
        with tempfile.TemporaryDirectory() as tmp:
            probe=SamplerDiagnostic(o,targets,Path(tmp),1)
            with self.assertRaises(RuntimeError):probe(x,b)
            self.assertEqual(probe.seen,set())

    def test_mutated_rng_state_fails_and_keeps_diagnostic_evidence(self):
        import json
        o,b,x=setup(torch.bfloat16)
        class BadNative:
            sampling_states=N(seeds=N(gpu=torch.full((4,),42)))
            def __call__(self,logits,batch):
                self.sampling_states.seeds.gpu.add_(1)
                return N(sampled_token_ids=logits.argmax(-1).reshape(-1,1),
                    num_sampled=torch.ones(4,dtype=torch.int32),num_rejected=torch.zeros(4,dtype=torch.int32))
        o.original_sampler=BadNative();o.active={f'q{i}':i for i in range(4)}
        o.runner.req_states=N(req_id_to_index=o.active);o.runner.steer_vector_state._dynamic_indices=dict(o.active)
        b.positions=torch.tensor([1,29,49,39]);b.logits_indices=torch.arange(4)
        with tempfile.TemporaryDirectory() as tmp:
            probe=SamplerDiagnostic(o,{f'q{i}':{int(o.count[i])} for i in range(4)},Path(tmp),16*1024*1024)
            with self.assertRaisesRegex(RuntimeError,'invariant'):probe(x,b)
            log=json.loads((Path(tmp)/'checks.jsonl').read_text())
            self.assertFalse(log['passed']);self.assertFalse(log['seed_unchanged'])
            self.assertTrue((Path(tmp)/'capture_000.pt').exists());self.assertEqual(probe.seen,set())


if __name__=='__main__':unittest.main(verbosity=2)
