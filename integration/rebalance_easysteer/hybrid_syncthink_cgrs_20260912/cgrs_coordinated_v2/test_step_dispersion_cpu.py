import unittest,math
from step_dispersion_reference import StepDispersion
class Signals(unittest.TestCase):
    def signal(self,x):
        s=StepDispersion();s.accept(0,start=True)
        for p in x:s.accept(p)
        s.accept(0,boundary=True);return s
    def test_same_mean_different_dispersion(self):
        a=self.signal([.9,.9]).last;b=self.signal([.8,1.]).last
        self.assertAlmostEqual(a['mean'],b['mean']);self.assertAlmostEqual(a['variance'],0);self.assertAlmostEqual(b['variance'],.01)
    def test_boundary_exclusion_and_empty(self):
        s=self.signal([.8,1.]);previous=s.last.copy();s.accept(0,boundary=True)
        self.assertEqual(s.last,previous);s.accept(.4);s.accept(0,boundary=True);self.assertEqual(s.last['count'],1)
    def test_end_and_new_request(self):
        s=self.signal([.8]);s.accept(0,end=True);s.accept(.2);self.assertEqual(s.n,0);self.assertIsNone(s.last)
        s.accept(0,start=True);self.assertEqual(s.n,0)
    def test_nonfinite_rejected(self):
        s=self.signal([.8])
        for p in [math.nan,math.inf,-.1,1.1]:
            with self.assertRaises(ValueError):s.accept(p)
if __name__=='__main__':unittest.main()
