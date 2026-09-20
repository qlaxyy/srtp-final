import unittest
from batch_layout_diagnostic import validate_input_slice,ready_for_refill,logical_geometry

class LayoutCPU(unittest.TestCase):
    def test_prompt_decode_boundary(self):
        validate_input_slice([11,12],[13,14],[1,2],[12,13])
        for pos,tok in [([2],[12]),([-1],[14]),([4],[14]),([0,1],[11])]:
            with self.assertRaises(RuntimeError):validate_input_slice([11,12],[13,14],pos,tok)

    def test_refill_waits_for_every_first_request(self):
        self.assertFalse(ready_for_refill({'a':33,'b':31},['a','b']))
        self.assertFalse(ready_for_refill({'a':33},['a','b']))
        self.assertFalse(ready_for_refill({},[]))
        self.assertTrue(ready_for_refill({'a':32,'b':32},['a','b']))

    def test_geometry_ignores_identity_not_execution_shape(self):
        def g(rid,slot,n):return [dict(num_tokens=n,padded_tokens=8,members=[dict(request_id=rid,slot=slot,row=0,count=2)])]
        self.assertEqual(logical_geometry(g('a',1,1)),logical_geometry(g('b',3,1)))
        self.assertNotEqual(logical_geometry(g('a',1,1)),logical_geometry(g('a',1,2)))

if __name__=='__main__':unittest.main(verbosity=2)
