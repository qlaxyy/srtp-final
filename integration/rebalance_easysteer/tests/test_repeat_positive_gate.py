"""CPU contract tests: causal repeat evidence and applied scales across KV replay.

The small NumPy state double checks hook ownership/timing, not torch/CUDA kernels.
"""
import ast
import contextlib
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'eval'))
from repeat_positive_gate import CompleteStepRepeat, RepeatPositiveAdapter
from repeat_positive_gate import applied_coefficient, byte_pieces, install_repeat_gate


class Tensor(np.ndarray):
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return self.copy()
    def clamp(self, max): return np.minimum(self, max)


def tensor(value):
    return np.asarray(value).view(Tensor)


class StateDouble:
    supports_kv_replay = True

    def __init__(self):
        self._history = tensor(np.zeros((3, 100)))
        self._coefs = tensor(np.zeros(3))
        self._step_prob_sum = np.zeros(3)
        self._step_tok_count = np.zeros(3, dtype=int)
        self._prev_step_mean = np.full(3, np.nan)
        self._in_think = np.zeros(3, dtype=bool)
        self._dynamic_params, self._dynamic_indices = {}, {}
        self._prompt_lengths, self._history_lengths = {}, {}
        self.saved = {}
        self.observe_calls = 0

    def has_dynamic(self): return bool(self._dynamic_params)

    def add_request(self, req_id, request, manager, *, req_index,
                    prompt_token_ids, num_generated_tokens=0):
        if request is None: return
        self._dynamic_params[req_id] = SimpleNamespace(boundary_token_ids=(9,),
            think_start_token_id=7, think_end_token_id=8)
        self._dynamic_indices[req_id] = req_index
        self._prompt_lengths[req_id] = len(prompt_token_ids)
        self._history_lengths[req_id] = len(prompt_token_ids)
        self._in_think[req_index] = 7 in prompt_token_ids
        if req_id in self.saved:
            fields, hist, length = self.saved.pop(req_id)
            for name, value in fields.items(): getattr(self, name)[req_index] = value
            self._history[req_index] = hist
            self._history_lengths[req_id] = length
            assert length == len(prompt_token_ids)+num_generated_tokens
        elif num_generated_tokens:
            raise RuntimeError('Missing saved history')

    def fields(self):
        return ('_coefs', '_step_prob_sum', '_step_tok_count', '_prev_step_mean', '_in_think')

    def remove_request(self, req_id, manager):
        idx = self._dynamic_indices.pop(req_id, None)
        self._dynamic_params.pop(req_id, None)
        self._prompt_lengths.pop(req_id, None)
        self._history_lengths.pop(req_id, None)
        if idx is not None:
            self._history[idx] = 0
            for field in self.fields(): getattr(self, field)[idx] = 0

    def suspend_request(self, req_id):
        idx = self._dynamic_indices[req_id]
        self.saved[req_id] = ({k: getattr(self, k)[idx].copy() for k in self.fields()},
            self._history[idx].copy(), self._history_lengths[req_id])

    def discard_suspended(self, req_id): self.saved.pop(req_id, None)

    def observe_sample(self, batch, tokens, probabilities):
        self.observe_calls += 1
        for pos, req in enumerate(batch.req_ids):
            if req not in self._dynamic_params: continue
            if batch.num_computed_tokens_np[pos]+batch.num_scheduled_tokens[pos] < batch.prefill_len_np[pos]: continue
            idx = self._dynamic_indices[req]
            token = int(tokens[pos, 0])
            if token == 8: self._in_think[idx] = False
            if token != 9:
                self._step_prob_sum[idx] += probabilities[pos]
                self._step_tok_count[idx] += 1
            elif self._step_tok_count[idx]:
                c = self._step_prob_sum[idx]/self._step_tok_count[idx]
                self._coefs[idx] = c-.5  # Test law makes the sign controllable.
                self._prev_step_mean[idx] = c
                self._step_prob_sum[idx] = self._step_tok_count[idx] = 0
            col = batch.num_computed_tokens_np[pos]+batch.num_scheduled_tokens[pos]
            self._history[idx, col] = self._coefs[idx]*self._in_think[idx]
            self._history_lengths[req] = col+1


PIECES = {1:b'A', 2:b'B', 3:b'new value', 7:b'<think>', 8:b'</think>', 9:b'\n\n'}
REQUEST = SimpleNamespace(algorithm='rebalance', rebalance_paper_parameters=None,
                          rebalance_boundary_token_ids=[9])


def sample(state, reqs, tokens, probabilities=None, partial=False):
    batch = SimpleNamespace(req_ids=reqs,
        num_computed_tokens_np=[0 if partial else state._history_lengths[r]-1 for r in reqs],
        num_scheduled_tokens=[1]*len(reqs),
        prefill_len_np=[state._prompt_lengths[r] for r in reqs])
    state.observe_sample(batch, tensor([[t] for t in tokens]),
                         tensor(probabilities or [.8]*len(reqs)))


class CompleteStepTests(unittest.TestCase):
    def test_complete_blocks_only_and_mismatch_releases_gate(self):
        m = CompleteStepRepeat([9],7,8)
        for token in [1,9,2,9,1,9,2]: self.assertIsNone(m.observe(token,PIECES[token]))
        event = m.observe(9,PIECES[9])
        self.assertEqual((event['period'],event['detected_token']),(2,8))
        saved = dict(event)
        self.assertIsNone(m.observe(3,PIECES[3]))
        self.assertIsNone(m.observe(9,PIECES[9]))
        self.assertEqual(event,saved)

    def test_formula_fragments_numbers_and_symbols_do_not_match_whole_steps(self):
        for a,b in [(b'x=1',b'x=2'),(b'x+1',b'x-1'),(b'a a',b'b'),(b'f(f(2))',b'f(2)')]:
            m=CompleteStepRepeat([9],7,8)
            for text in [a,b]:
                self.assertIsNone(m.observe(1,text))
                self.assertIsNone(m.observe(9,b'\n\n'))

    def test_tokenization_can_differ_but_boundary_symbols_are_retained(self):
        m=CompleteStepRepeat([9],7,8)
        m.observe(1,b' value=1');m.observe(9,b'.\n\n')
        for part in [b'value',b'=',b'1']: m.observe(2,part)
        self.assertIsNotNone(m.observe(9,b'.\n\n'))
        m.observe(1,b'value=1')
        self.assertIsNone(m.observe(9,b'?\n\n'))

    def test_eof_end_of_thinking_and_unready_boundaries_never_act(self):
        m=CompleteStepRepeat([9],7,8)
        m.observe(1,b'A');m.observe(9,b'\n\n');m.observe(1,b'A')
        self.assertIsNone(m.observe(8,b'</think>'))
        self.assertIsNone(m.observe(9,b'\n\n'))
        m=CompleteStepRepeat([9],7,8)
        m.observe(9,b'.\n\n')
        event=m.observe(9,b'.\n\n')
        self.assertFalse(event['eligible_boundary'])

    def test_snapshot_retains_partial_bytes_without_sharing_mutable_state(self):
        m=CompleteStepRepeat([9],7,8)
        m.observe(1,b'\xe5'); saved=m.snapshot()
        m.observe(2,b'\x80\xbc');m.observe(9,b'\n\n')
        self.assertEqual(saved.pending,b'\xe5')
        saved.observe(2,b'\x80\xbc');saved.observe(9,b'\n\n')
        self.assertEqual(m.units,saved.units)

    def test_repetition_matches_independent_suffix_oracle(self):
        rng=random.Random(20260911)
        for _ in range(25):
            m=CompleteStepRepeat([9],7,8); words=[]
            for _ in range(70):
                word=rng.choice([b'A',b'B',b'C']);words.append(word)
                m.observe(1,word);event=m.observe(9,b'\n\n')
                oracle=any(words[-2*p:-p]==words[-p:] for p in range(1,len(words)//2+1))
                self.assertEqual(event is not None,oracle)


class RuntimeContractTests(unittest.TestCase):
    def create(self, mode):
        state=StateDouble();adapter=RepeatPositiveAdapter(state,PIECES,mode)
        state.add_request('r',REQUEST,None,req_index=0,prompt_token_ids=[7])
        return state,adapter

    def test_cancel_changes_only_current_applied_positive_scale(self):
        baseline,shadow=self.create('shadow');candidate,gate=self.create('cancel_positive')
        for i,token in enumerate([1,9,2,9,1,9,2,9,3,9],1):
            for state in [baseline,candidate]: sample(state,['r'],[token])
            for field in baseline.fields():
                np.testing.assert_equal(getattr(baseline,field),getattr(candidate,field))
            if i==8:
                self.assertAlmostEqual(baseline._history[0,i],.3)
                self.assertEqual(candidate._history[0,i],0)
        self.assertEqual(candidate.observe_calls,10)
        self.assertEqual(gate.export()['changed_scales'],1)
        self.assertEqual(shadow.export()['changed_scales'],0)
        self.assertAlmostEqual(candidate._history[0,10],.3)

    def test_negative_and_zero_coefficients_remain_unchanged(self):
        for mode in ('off','shadow','cancel_positive'):
            for coefficient in (-1.5,0.,.1,1e-9):
                expected=min(coefficient,0.) if mode=='cancel_positive' else coefficient
                self.assertEqual(applied_coefficient(coefficient,True,mode),expected)
                self.assertEqual(applied_coefficient(coefficient,False,mode),coefficient)
        state,gate=self.create('cancel_positive')
        for token in [1,9,1,9]: sample(state,['r'],[token],[.2])
        self.assertAlmostEqual(state._history[0,4],-.3)
        self.assertEqual(gate.export()['changed_scales'],0)

    def test_request_order_and_slot_reuse_do_not_share_repetition(self):
        state,gate=self.create('cancel_positive')
        state.add_request('s',REQUEST,None,req_index=2,prompt_token_ids=[7])
        for n,(r,s) in enumerate(zip([1,9,1,9],[2,9,3,9])):
            reqs,tokens=(['r','s'],[r,s]) if n%2==0 else (['s','r'],[s,r])
            sample(state,reqs,tokens)
        self.assertEqual(state._history[0,4],0)
        self.assertAlmostEqual(state._history[2,4],.3)
        state.remove_request('r',None)
        state.add_request('new',REQUEST,None,req_index=0,prompt_token_ids=[7])
        sample(state,['new'],[1]);sample(state,['new'],[9])
        self.assertAlmostEqual(state._history[0,2],.3)

    def test_preemption_restores_monitor_controller_and_applied_history(self):
        state,gate=self.create('cancel_positive')
        for token in [1,9,1,9,2]: sample(state,['r'],[token])
        state.suspend_request('r');state.remove_request('r',None)
        state.add_request('r',REQUEST,None,req_index=2,prompt_token_ids=[7],num_generated_tokens=5)
        self.assertEqual(state._history[2,4],0)
        self.assertEqual(gate.live['r'].pending,b'B')
        for token in [9,2,9]: sample(state,['r'],[token])
        self.assertEqual(state._history[2,8],0)
        self.assertEqual(gate.export()['changed_scales'],2)

    def test_partial_prefill_is_not_counted_as_generated_text(self):
        state=StateDouble();gate=RepeatPositiveAdapter(state,PIECES,'cancel_positive')
        state.add_request('r',REQUEST,None,req_index=0,prompt_token_ids=[7,1,2,3])
        sample(state,['r'],[1],partial=True)
        self.assertEqual(gate.live['r'].observed_tokens,0)
        self.assertEqual(gate.sample_batches_copied,0)

    def test_disabled_mode_leaves_original_methods_and_no_tokenizer_io(self):
        state=StateDouble(); original=state.observe_sample
        self.assertIsNone(install_repeat_gate(state,None,'off'))
        self.assertEqual(state.observe_sample,original)
        self.assertNotIn('_repeat_positive_adapter',state.__dict__)

    def test_wrong_tokenizer_fails_before_request_admission(self):
        state=StateDouble()
        RepeatPositiveAdapter(state,PIECES | {9:b'not a boundary'},'shadow')
        with self.assertRaisesRegex(RuntimeError,'tokenizer'):
            state.add_request('r',REQUEST,None,req_index=0,prompt_token_ids=[7])
        self.assertFalse(state.has_dynamic())

    def test_evaluator_rejects_accidental_pair_or_wrong_token_budget(self):
        import argparse
        tree=ast.parse((BASE/'eval/rebalance_dynamic_eval.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        ns=dict(argparse=argparse,Path=Path,DEFAULT_MODEL='model',DEFAULT_DATASET='data',
                DEFAULT_VECTOR='vector',DEFAULT_OUTPUT='new.json')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'parse_args','exec'),ns)
        valid=['--repeat-gate','cancel_positive','--diagnostic-group','rebalance_dynamic',
               '--max-tokens','16000','--calibration-fit','fit.json']
        with patch.object(sys,'argv',['eval',*valid]): self.assertEqual(ns['parse_args']().repeat_gate,'cancel_positive')
        for invalid in [valid[:2],valid[:-2],valid[:-4]+['--max-tokens','4000','--calibration-fit','fit.json']]:
            with patch.object(sys,'argv',['eval',*invalid]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): ns['parse_args']()


if __name__=='__main__': unittest.main()
