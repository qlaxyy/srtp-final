import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engineering import sha
from narrow_grade import decision
from narrow_runner import cases, validate_engineering_gate, validate_prompt_capacity


class HistoryScreenTests(unittest.TestCase):
    def test_long_prompt_rejected_before_loading_and_equal_capacity_allowed(self):
        with self.assertRaises(AssertionError):validate_prompt_capacity([[1]*1800],16000,17408)
        self.assertEqual(validate_prompt_capacity([[1]*1920],16000,17920),1920)

    def test_only_screen_and_only_candidate_gets_history_gate(self):
        plan={'candidate_kind':'first_reflection_screen'}
        self.assertEqual(cases(plan,'screen'),[
            ('R','off','original14','none'),('RC14','negative','original14','none'),
            ('RChistory','negative','original14','after_first_reflection')])
        for phase in ('engineering','speed'):
            with self.assertRaises(AssertionError):cases(plan,phase)

    def test_old_gate_reuse_requires_exact_mechanism_and_plan_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=root/'gate.json';source=root/'adapter.py'
            source.write_text('unchanged\n')
            gate.write_text(json.dumps(dict(passed=True,plan_sha256='prior')))
            plan=dict(candidate_kind='first_reflection_screen',engineering_evidence=dict(
                gate_sha256=sha(gate),plan_sha256='prior',
                unchanged_mechanism_sources={'adapter.py':sha(source,True)}))
            with patch('narrow_runner.ROOT',root):
                validate_engineering_gate(plan,None,gate)
                source.write_text('changed\n')
                with self.assertRaises(AssertionError):validate_engineering_gate(plan,None,gate)
                source.write_text('unchanged\n');plan['engineering_evidence']['plan_sha256']='other'
                with self.assertRaises(AssertionError):validate_engineering_gate(plan,None,gate)

    def test_baseline_identical_candidate_cannot_promote(self):
        groups={k:dict(n=100,correct=c,mean_total_tokens=t,mean_thinking_tokens=t-100,capped=0)
                for k,c,t in [('R',93,2600),('RC14',91,2500),('RChistory',93,2600)]}
        self.assertFalse(decision(groups,'RChistory')['promote'])
        groups['RChistory']['mean_total_tokens']=2550
        groups['RChistory']['mean_thinking_tokens']=2450
        self.assertTrue(decision(groups,'RChistory')['promote'])


if __name__=='__main__':unittest.main(verbosity=2)
