"""Guard mismatched slots and accidental full-effect scope at cheapest CPU layer."""
import unittest
from sampler_diagnostic import capture_positions,verify_mapping


class DiagnosticProtocolTests(unittest.TestCase):
    def test_capture_boundary_rejects_negative_or_unbounded_windows(self):
        self.assertEqual(capture_positions(6),[0,5,6,127])
        self.assertEqual(capture_positions(0),[0,127])
        for value in (-1,128,None,True):
            with self.assertRaises(ValueError):capture_positions(value)

    def test_reordered_batch_routes_by_slot_not_iteration_order(self):
        active={'b':7,'a':2};self.assertEqual(verify_mapping(active,active,active,[2,7]),['a','b'])
        for model,control,batch in [({'b':2,'a':7},active,[2,7]),(active,{'b':7,'a':3},[2,7]),
                                    (active,active,[2,2]),(active,active,[9])]:
            with self.assertRaises(RuntimeError):verify_mapping(active,model,control,batch)
        with self.assertRaises(RuntimeError):verify_mapping({'a':2,'b':2},active,active,[2])


if __name__=='__main__':unittest.main(verbosity=2)
