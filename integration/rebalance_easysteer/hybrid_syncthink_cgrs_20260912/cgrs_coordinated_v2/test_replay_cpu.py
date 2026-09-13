from types import SimpleNamespace as N
import unittest
import numpy as np
from replay_adapter import ReplayAdapter


def fixture():
    o=ReplayAdapter.__new__(ReplayAdapter);o.active={};o.completed={};o.replay_events=[]
    saved=dict(opening=1,thinking=1,count=32,prompt_len=10,eligible_count=4,changed_count=4,first_change=15)
    o.suspended={'q':saved}
    o.runner=N(req_states=N(req_id_to_index={'q':3}),steer_vector_state=N(_dynamic_indices={'q':3}))
    o.original_add=lambda output:None
    for key in o.fields:setattr(o,key,np.zeros(4,dtype=np.int64))
    params=N(n=1,structured_outputs=None,logit_bias=None,bad_words=None,presence_penalty=0,frequency_penalty=0,repetition_penalty=1,min_tokens=0)
    req=N(req_id='q',sampling_params=params,num_computed_tokens=0,prompt_token_ids=[151648]*10,prefill_token_ids=[1]*42)
    return o,N(scheduled_new_reqs=[req]),saved


class Tests(unittest.TestCase):
    def test_reassigned_slot_restores_all_lexical_state(self):
        o,out,saved=fixture();o.add_requests(out)
        self.assertEqual({k:int(getattr(o,k)[3]) for k in o.fields},saved)
        self.assertEqual(o.active,{'q':3});self.assertFalse(o.suspended)
    def test_replay_clock_mismatch_rejected_before_native_mutation(self):
        o,out,_=fixture();out.scheduled_new_reqs[0].prefill_token_ids.pop()
        with self.assertRaisesRegex(RuntimeError,'clock mismatch'):o.add_requests(out)
        self.assertFalse(o.active)
    def test_generated_prefix_without_saved_state_rejected(self):
        o,out,_=fixture();o.suspended={}
        with self.assertRaisesRegex(RuntimeError,'lacks lexical'):o.add_requests(out)
    def test_fresh_slot_starts_clean(self):
        o,out,_=fixture();o.suspended={};out.scheduled_new_reqs[0].prefill_token_ids=[1]*10
        o.add_requests(out)
        self.assertEqual(int(o.count[3]),0);self.assertEqual(int(o.thinking[3]),1)
        self.assertEqual(int(o.opening[3]),0);self.assertEqual(int(o.first_change[3]),-1)


if __name__=='__main__':unittest.main(verbosity=2)
