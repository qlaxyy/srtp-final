import math
import unittest
from audit_strength_coupling import penalty


class StrengthTests(unittest.TestCase):
    def test_endpoints_and_continuity(self):
        self.assertEqual(penalty(0,-1.5),0)
        self.assertEqual(penalty(-1.5,-1.5),math.log(2))
        self.assertAlmostEqual(penalty(-.75,-1.5),math.log(2)/2)
        self.assertLess(penalty(-1e-8,-1.5),1e-8)
        self.assertEqual(penalty(-1e-8,-1.5,'fixed'),math.log(2))

    def test_bounds_and_invalids(self):
        self.assertEqual(penalty(-4,-1.5),math.log(2))
        for x in (float('nan'),float('inf'),-float('inf'),.1):
            self.assertEqual(penalty(x,-1.5),0)
        for bound in (0,1,float('nan')):
            with self.assertRaises(ValueError):penalty(-1,bound)

    def test_monotonic_and_rescaling(self):
        a=[penalty(-x/100,-1) for x in range(101)]
        self.assertEqual(a,sorted(a))
        self.assertAlmostEqual(penalty(-.3,-1.5),penalty(-.6,-3))


if __name__=='__main__':unittest.main()
