"""Meaningful geometry, finite precision, chronology and asset guard checks."""
import sys
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mechanism_candidates import directions, moments, bfloat16, zero_loss_upper, checkpoint_cost, read_vector,sha
from audit_mechanism_candidates import proxy_diagnostic
from audit_control_points import map_positions
from audit_followup_candidates import clip_negative


class MechanismTests(unittest.TestCase):
    def test_latent_clip_preserves_positive_and_never_crosses_negative_center(self):
        c=np.array([-2.,-2.,-2.,.1,0.])
        s=np.array([-1.,.5,4.,-1.,3.])
        result=clip_negative(c,s,0.,1.)
        np.testing.assert_array_equal(result,[0.,-.5,-2.,.1,0.])
        self.assertTrue(np.all(s[1:3]+result[1:3]>=0))
        self.assertTrue(np.all(np.abs(result)<=np.abs(c)))
        with self.assertRaises(ValueError):clip_negative(c,s,0.,-1.)

    def test_control_points_exclude_prompt_and_follow_adjacent_delimiters(self):
        row=dict(prompt_token_ids=[99,98],token_ids=[1,2,9,9,3,4,9])
        steps=[dict(start=0,stop=2),dict(start=4,stop=6)]
        saved=dict(question=0,positions=[2,6],think_stop=7)
        mapped=map_positions(row,steps,saved,{9})
        self.assertEqual(mapped['predecessor'],[1,5])
        self.assertEqual(mapped['generated_predecessor'],[False,True])

    def test_control_points_reject_stale_offsets_and_nondelimiter_predecessor(self):
        row=dict(prompt_token_ids=[99],token_ids=[1,2,3,9])
        saved=dict(question=0,positions=[2],think_stop=4)
        with self.assertRaisesRegex(ValueError,'not a delimiter'):
            map_positions(row,[dict(start=1,stop=3)],saved,{9})
        with self.assertRaisesRegex(ValueError,'positions differ'):
            map_positions(row,[dict(start=0,stop=3)],saved,{9})

    def test_minimum_displacement_constraint_and_optimality(self):
        a=dict(count=7,mean=np.array([3.,1.,20.]),variance=np.array([2.,8.,1.]))
        b=dict(count=4,mean=np.array([1.,0.,20.]),variance=np.array([3.,1.,2.]))
        fit=directions(a,b); w=fit['w']; target=fit['score']; v=fit['min_displacement']
        self.assertAlmostEqual(w@v,target)
        rng=np.random.default_rng(42)
        for _ in range(20):
            r=rng.normal(size=3); r-=w*(w@r)/(w@w)
            self.assertLessEqual(np.linalg.norm(v),np.linalg.norm(v+r)+1e-12)
        r=fit['orthogonal_mean']
        self.assertAlmostEqual(fit['u']@r,0)
        self.assertAlmostEqual(w@r,target)

    def test_chunk_moments_match_direct_population(self):
        x=np.random.default_rng(7).normal(size=(93,8)).astype(np.float32)
        mask=np.arange(len(x))%3!=0
        out=moments(x,{'a':mask},chunk=11)['a']
        np.testing.assert_allclose(out['mean'],x[mask].astype(float).mean(0),rtol=0,atol=1e-14)
        np.testing.assert_allclose(out['variance'],x[mask].astype(float).var(0),rtol=0,atol=1e-14)

    def test_bfloat16_ties_even_and_representable_values(self):
        values=np.array([0.,1.,-2.,1.00390625,1.01171875],dtype=np.float32)
        np.testing.assert_array_equal(bfloat16(values),[0.,1.,-2.,1.,1.015625])
        with self.assertRaises(ValueError): bfloat16([np.nan])

    def test_proxy_never_borrows_future_or_another_question(self):
        values=[.2,.2,.8,.8,.8,.8]
        steps=[dict(question=0,start=0,stop=2,confidence=.2),dict(question=0,start=2,stop=6,confidence=.8)]
        rows=[dict(logprobs=np.log(values).tolist())]
        # A nonstationary step: the current first half is exact; borrowing is harmful.
        # Avoid zero denominator by making the target slightly different.
        rows[0]['logprobs']=np.log([.2,.2,.7,.7,.9,.9]).tolist()
        result=proxy_diagnostic(steps,rows,{0})
        json.dumps(result,allow_nan=False)
        self.assertGreater(result['short']['ratio'],1)
        self.assertEqual(result['short']['steps'],1)

    def test_cost_and_small_sample_risk(self):
        self.assertEqual(checkpoint_cost(1024)['checkpoints'],0)
        self.assertEqual(checkpoint_cost(16000)['max_extra_probe_tokens'],720)
        self.assertGreater(zero_loss_upper(100),.029)
        self.assertLess(zero_loss_upper(299),.01)

    def test_untrusted_tensor_template_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.pt'; p.write_bytes(b'not a trusted tensor')
            with self.assertRaises(ValueError): read_vector(p,p)

    def test_hashing_does_not_require_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'data';p.write_bytes(b'a'*(1024*1024+19))
            with patch.object(hashlib,'file_digest',None):
                self.assertEqual(sha(p),hashlib.sha256(p.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
