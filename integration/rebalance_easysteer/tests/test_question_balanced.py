"""Statistical invariants for question-balanced population moments, CPU only."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_question_balanced import moments, geometry


class MomentsTests(unittest.TestCase):
    def test_matches_explicit_state_weights_with_missing_question(self):
        x=np.array([[1.,2.],[3.,8.],[10.,5.]])
        m,v,n=moments(x,np.array([0,0,2]))
        w=np.array([.25,.25,.5])
        np.testing.assert_allclose(m,w@x)
        np.testing.assert_allclose(v,w@((x-m)**2))
        self.assertEqual(n,2)

    def test_duplicating_all_states_in_one_question_does_not_reweight_it(self):
        x=np.array([[1.,2.],[3.,8.],[10.,5.]])
        a=moments(x,np.array([0,0,2]))
        b=moments(np.concatenate([x,x[:2],x[:2]]),np.array([0,0,2,0,0,0,0]))
        np.testing.assert_allclose(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1])

    def test_equal_class_counts_reproduce_step_uniform_moments(self):
        x=np.arange(24,dtype=float).reshape(8,3)
        m,v,_=moments(x,np.repeat([0,1,2,3],2))
        np.testing.assert_allclose(m,x.mean(0))
        np.testing.assert_allclose(v,x.var(0))

    def test_raw_difference_mean_crossing_and_maximum(self):
        over=np.array([[2.,4.],[3.,1.],[5.,2.]])
        under=np.array([[-1.,1.],[0.,0.]])
        _,m,a,res=geometry(over.mean(0),under.mean(0),over.var(0),under.var(0),over)
        self.assertAlmostEqual(m,.5)
        self.assertGreaterEqual(a,m)
        self.assertLess(abs(res),1e-10)


if __name__=='__main__':
    unittest.main()
