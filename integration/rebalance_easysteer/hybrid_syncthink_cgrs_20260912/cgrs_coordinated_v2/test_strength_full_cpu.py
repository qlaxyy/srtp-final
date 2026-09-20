import copy,unittest
from engineering import HERE,read
from strength_full import validate
from narrow_runner import cases,validate_engineering_gate
class FullScope(unittest.TestCase):
    def setUp(self):
        self.p=read(HERE/'strength_fullmath500_20260917/plan.json')
        self.r=dict(scope='strength_1p5b_math500_scaled_only',authorized_phases=['full'],exposed_test_reuse_acknowledged=True)
    def test_scope_and_engineering(self):
        validate(self.p,self.r)
        self.assertEqual(cases(self.p,'full'),[('math_test','negative','original14','none')])
        validate_engineering_gate(self.p,None,HERE/'strength_screen100_20260917/engineering_gate.json')
    def test_reject_expansion_and_changed_penalty(self):
        for key,value in [('arms',['RCscaled','RCconstant']),('runtime',dict(self.p['runtime'],async_scheduling=True)),('penalty',dict(self.p['penalty'],lower_bound=-1.))]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(AssertionError):validate(p,self.r)
        self.p['datasets']['math_test']['rows'].pop()
        with self.assertRaises(AssertionError):validate(self.p,self.r)
if __name__=='__main__':unittest.main()
