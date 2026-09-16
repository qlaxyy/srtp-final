"""Real Torch CPU hook contract, required before any model loading."""
import tempfile,unittest
from types import SimpleNamespace as N
import numpy as np
import torch
from batch_layout_diagnostic import FrozenPrefixTrace,STATE_FIELDS

def fixture(folder,reference=None):
    state=N(_dynamic_indices={'q':1},_history_lengths={'q':2},_history=torch.zeros(2,8))
    for f in STATE_FIELDS:setattr(state,f,torch.zeros(2))
    state._prev_step_mean[1]=float('nan')
    state.token_scales=lambda b:state._history[1,b.positions]
    observed=[]
    state.observe_sample=lambda b,t,p:observed.append((t.clone(),p.clone()))
    native=lambda x,b:N(sampled_token_ids=x.argmax(-1).reshape(-1,1))
    runner=N(sampler=native,steer_vector_state=state,req_states=N(req_id_to_index={'q':1}))
    scheduler=N(_preempt_request=lambda:None)
    llm=N(llm_engine=N(engine_core=N(engine_core=N(model_executor=N(driver_worker=N(worker=N(model_runner=runner))),scheduler=scheduler))))
    rows=[dict(prompt_token_ids=[11,12],forced_token_prefix=[13]*128,capture_positions=[0])]
    trace=FrozenPrefixTrace(llm,folder,rows,reference);trace.register(['q'],[0])
    batch=N(idx_mapping=torch.tensor([1]),num_reqs=1,query_start_loc=torch.tensor([0,2]),positions=torch.tensor([0,1]),input_ids=torch.tensor([11,12]),num_tokens=2,num_tokens_after_padding=2,req_ids=['q'],num_computed_tokens_np=np.array([0]),num_scheduled_tokens=np.array([2]),prefill_len_np=np.array([2]),num_draft_tokens=0,seq_lens=torch.tensor([2]))
    return trace,batch,state,observed,runner,native

class NativeLayout(unittest.TestCase):
    def test_forced_token_is_observed_and_hooks_restore(self):
        with tempfile.TemporaryDirectory() as d:
            t,b,s,observed,runner,native=fixture(d)
            try:
                t.token_scales(b)
                output=t(torch.tensor([[0.,1.]]),b)
                self.assertEqual(t.native_draws[(0,0)],1)
                t.observe_sample(b,output.sampled_token_ids,torch.tensor([.6]))
                self.assertEqual(observed[0][0].item(),13)
                self.assertEqual(t.counts['q'],1)
            finally:t.close()
            self.assertIs(runner.sampler,native)

    def test_reference_restores_nan_scalars_and_history_before_forward(self):
        with tempfile.TemporaryDirectory() as d:
            t,b,s,_,_,_=fixture(d)
            snap=t.snapshot('q',1);t.reference={'states':{(0,0):snap}}
            s._coefs[1]=9;s._history[1,:2]=7;s._prev_step_mean[1]=1
            try:
                scales=t.token_scales(b)
                self.assertTrue(torch.equal(scales,torch.zeros(2)))
                self.assertTrue(t.same_state(t.snapshot('q',1),snap))
                self.assertEqual(t.state_locks,1)
            finally:t.close()

    def test_invalid_input_or_slot_never_reaches_native_forward(self):
        with tempfile.TemporaryDirectory() as d:
            t,b,s,_,_,_=fixture(d)
            try:
                b.input_ids[1]=99
                with self.assertRaisesRegex(RuntimeError,'input differs'):t.token_scales(b)
                b.input_ids[1]=12;s._dynamic_indices['q']=0
                with self.assertRaisesRegex(RuntimeError,'slot disagreement'):t.token_scales(b)
                with self.assertRaises(RuntimeError):t.reject_preemption()
            finally:t.close()

if __name__=='__main__':unittest.main(verbosity=2)
