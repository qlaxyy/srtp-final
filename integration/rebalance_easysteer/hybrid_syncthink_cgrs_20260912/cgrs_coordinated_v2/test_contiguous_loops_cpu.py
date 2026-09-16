import unittest
from audit_contiguous_loops import first_loop

class LoopTests(unittest.TestCase):
    def test_exact_third_copy_and_future_independence(self):
        block=list(range(8));a=first_loop([99]+block*3)
        self.assertEqual((a['start'],a['end'],a['period']),(1,25,8))
        b=first_loop([99]+block*4+[99]);self.assertEqual(b['end'],a['end']);self.assertEqual(b['subsequent_exact_loop_tokens'],8)
    def test_near_match_and_low_diversity_rejected(self):
        self.assertIsNone(first_loop(list(range(8))*2+list(range(7))+[99]))
        self.assertIsNone(first_loop([1,2]*100))
    def test_no_early_trigger(self):
        for n in range(24):self.assertIsNone(first_loop((list(range(8))*3)[:n]))

if __name__=='__main__':unittest.main(verbosity=2)
