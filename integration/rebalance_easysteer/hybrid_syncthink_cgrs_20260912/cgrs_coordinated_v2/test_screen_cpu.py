"""Meaningful synthetic paired-outcome checks, no model imports."""
import copy
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from screen_grade import analyze
from screen import ARMS


def fixture():
    results={};grades={}
    for a in ARMS:
        records=[dict(train_index=i,problem_sha256=str(i),tokens=100,thinking_tokens=80,
                      finish_reason='stop') for i in range(64)]
        results[a]=dict(records=records,generation_seconds=1.,setup_seconds=0.,events={},
                        control_gpu_seconds=None,checkpoint_io_seconds=0.)
        grades[a]=[dict(train_index=i,problem_sha256=str(i),correct=i<50) for i in range(64)]
    return results,grades


class Checks(unittest.TestCase):
    def test_identical_output_does_not_pass_screen(self):
        r,g=fixture();a=analyze(r,g)
        self.assertFalse(a['primary_passes_point_screen'])
        self.assertEqual(a['interaction_RCalways_minus_R_minus_Clex_plus_U']['tokens']['ci95'],[0.,0.])

    def test_margin_and_caps_not_accuracy_alone(self):
        r,g=fixture()
        for row in r['RCnegative']['records']:row.update(tokens=90,thinking_tokens=70)
        g['RCnegative'][0]['correct']=False
        self.assertTrue(analyze(r,g)['primary_passes_point_screen'])
        g['RCnegative'][1]['correct']=False
        self.assertFalse(analyze(r,g)['primary_passes_point_screen'])
        g['RCnegative'][1]['correct']=True
        r['RCnegative']['records'][0]['finish_reason']='length'
        self.assertFalse(analyze(r,g)['primary_passes_point_screen'])

    def test_factorial_scale_and_no_diagnostic_cherry_pick(self):
        r,g=fixture()
        for a,value in [('U',100),('R',80),('Clex',90),('RCalways',60),('RCnegative',80)]:
            for row in r[a]['records']:row['tokens']=value
        a=analyze(r,g)
        self.assertEqual(a['interaction_RCalways_minus_R_minus_Clex_plus_U']['tokens']['mean_token_interaction'],-10.)
        self.assertFalse(a['primary_passes_point_screen'])

    def test_reordered_pair_is_rejected(self):
        r,g=fixture();r['RCnegative']['records'].reverse()
        with self.assertRaises(AssertionError):analyze(r,g)

    def test_no_authorization_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            p=subprocess.run([sys.executable,str(Path(__file__).with_name('screen.py')),
                '--plan','missing','--receipt','missing','--output-root',temp],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Explicit authorization',p.stderr)
            self.assertEqual(list(Path(temp).iterdir()),[])


if __name__=='__main__':unittest.main(verbosity=2)
