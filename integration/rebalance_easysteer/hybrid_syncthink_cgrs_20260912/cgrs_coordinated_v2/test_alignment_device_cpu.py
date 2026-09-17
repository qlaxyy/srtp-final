"""CPU Torch contract tests for instance hooks; not native GPU acceptance."""
import unittest
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import torch
from label_alignment_adapter import AlignmentAdapter
from lexicon_automaton import Automaton
from prepare_label_alignment import load


def owner(control=False):
    o=AlignmentAdapter.__new__(AlignmentAdapter)
    o.torch=torch;o.large_suppression=not control;o.lexical_control=control
    size=151650
    pieces=['','Let',' me',' check',' calculate','Wait','For',' ']
    machine=Automaton(['let me check','wait'],opening=not control)
    tr,hit=machine.compile_tokens(pieces)
    o.lex_transition=torch.zeros((len(machine.states),size),dtype=torch.int32)
    o.lex_hits=torch.zeros((len(machine.states),size),dtype=torch.bool)
    o.lex_transition[:,:len(pieces)]=torch.from_numpy(tr)
    o.lex_hits[:,:len(pieces)]=torch.from_numpy(hit)
    o.lex_final=torch.from_numpy(machine.final)
    o.lex_classes=torch.arange(size)
    o.lex_state=torch.zeros(3,dtype=torch.long)
    for name in ['lex_hit','lex_ready','closed_hit','lex_open']:
        setattr(o,name,torch.zeros(3,dtype=torch.bool))
    o.lex_changes=torch.zeros(3,dtype=torch.long)
    o.thinking=torch.ones(3,dtype=torch.bool)
    o.boundary=torch.zeros(size,dtype=torch.bool);o.boundary[271]=True
    o.clean=o.boundary.clone()
    o.ids=torch.arange(len(pieces));o.lex_candidates=o.lex_hits[:,o.ids]
    o.runner=N(steer_vector_state=N(_coefs=torch.tensor([-.5,.1,-.5]),
        _prev_step_mean=torch.tensor([.95,.99,.95]),_step_tok_count=torch.tensor([3,3,0])))
    return o


class DeviceTests(unittest.TestCase):
    def test_default_off_installs_nothing(self):
        llm=N();o=AlignmentAdapter(llm,None)
        self.assertFalse(o.enabled);self.assertEqual(vars(llm),{})
        o.close()

    def test_phrase_mask_and_slot_mapping(self):
        o=owner();idx=torch.tensor([2,0,1]);valid=torch.tensor([False,True,True])
        o.accept_lexical_sample(idx,torch.tensor([271,271,271]),valid)
        self.assertEqual(o.lex_open.tolist(),[True,True,False])
        mask=o.candidate_mask(idx,valid)
        self.assertTrue(mask[1,5]);self.assertFalse(mask[1,1]);self.assertFalse(mask[2].any())
        o.accept_lexical_sample(idx,torch.tensor([1,1,1]),valid)
        o.accept_lexical_sample(idx,torch.tensor([2,2,2]),valid)
        mask=o.candidate_mask(idx,valid)
        self.assertTrue(mask[1,3]);self.assertFalse(mask[1,4])
        o.accept_lexical_sample(idx,torch.tensor([3,3,3]),valid)
        self.assertFalse(o.lex_open[0]);self.assertEqual(o.lex_state[2],0)

    def test_think_end_resets(self):
        o=owner();idx=torch.tensor([0]);valid=torch.tensor([True])
        o.accept_lexical_sample(idx,torch.tensor([271]),valid)
        o.accept_lexical_sample(idx,torch.tensor([1]),valid)
        o.accept_lexical_sample(idx,torch.tensor([151649]),valid)
        self.assertFalse(o.lex_open[0]);self.assertEqual(o.lex_state[0],0)

    def test_full_step_word_wait_for_not_reflection(self):
        o=owner(True);idx=torch.tensor([0]);valid=torch.tensor([True])
        for token in [5,6,271]:o.accept_lexical_sample(idx,torch.tensor([token]),valid)
        self.assertTrue(o.lex_ready[0]);self.assertFalse(o.closed_hit[0])
        for token in [5,271]:o.accept_lexical_sample(idx,torch.tensor([token]),valid)
        self.assertTrue(o.closed_hit[0]);self.assertFalse(o.lex_hit[0])

    def test_empty_step_does_not_update(self):
        o=owner(True);idx=torch.tensor([2]);valid=torch.tensor([True])
        o.accept_lexical_sample(idx,torch.tensor([271]),valid)
        self.assertFalse(o.lex_ready[2])

    def test_record_after_replacement(self):
        o=owner(True)
        root=Path(__file__).resolve().parents[4]
        runtime=load('alignment_device_runtime',root.parent/'srtp-final/sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py')
        params=runtime.ReBalanceParams(boundary_token_ids=(271,),think_start_token_id=151648,
            think_end_token_id=151649,q25c=.2,q75c=.8,low_val_1=-.5,low_val_2=-1.5,high_val_2=.1,curve_tau=.01)
        o.rstate=o.runner.steer_vector_state;o.control_runtime=runtime
        o.rstate._dynamic_params={'a':params,'b':params}
        o.lex_ready[:2]=torch.tensor([True,False]);o.closed_hit[0]=True
        recorded=[];o.original_record=lambda *args:recorded.append(o.rstate._coefs.clone())
        o.record_scales(N(req_ids=['a','b']),[0,1],torch.tensor([0,1]),torch.tensor([0,1]))
        self.assertAlmostEqual(float(recorded[0][0]),-1.5)
        self.assertAlmostEqual(float(recorded[0][1]),.1)
        self.assertEqual(o.lex_changes.tolist(),[1,0,0])


if __name__=='__main__':unittest.main()
