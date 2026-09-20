import copy
import unittest
from engineering import HERE,read
from narrow_runner import cases,validate_receipt
from strength_engineering import validate,adapter_options,check_result


class StrengthEngineeringTests(unittest.TestCase):
    def setUp(self):self.plan=read(HERE/'strength_engineering8_20260916/plan_run2.json')
    def test_fixed_batch_and_no_formal_run(self):
        validate(self.plan);self.assertEqual(len(cases(self.plan,'engineering')),7)
        for phase in ('screen','full','speed'):
            with self.assertRaises(AssertionError):cases(self.plan,phase)
    def test_scope_and_constant_tampering_rejected(self):
        for key,value in [('engineering_cap',16000),('model_family','7b')]:
            p=copy.deepcopy(self.plan);p[key]=value
            with self.assertRaises(AssertionError):validate(p)
        self.plan['penalty']['constant_scale']=.5
        with self.assertRaises(AssertionError):validate(self.plan)
    def test_unapproved_template_rejected(self):
        receipt=read(HERE/'strength_engineering8_20260916/receipt_template.json')
        with self.assertRaises(AssertionError):validate_receipt(self.plan,receipt,HERE/'strength_engineering8_20260916/plan.json')
    def test_arm_options_and_equivalence_check(self):
        self.assertEqual(adapter_options(self.plan,'R'),{})
        self.assertEqual(adapter_options(self.plan,'RCscaled')['penalty_mode'],'coefficient_scaled')
        r={'records':[{'token_ids':[1], 'R_history_sha256':'x'}]}
        check_result('Roff_scaled',{'R':r,'Roff_scaled':copy.deepcopy(r)})
        changed=copy.deepcopy(r);changed['records'][0]['R_history_sha256']='y'
        with self.assertRaises(AssertionError):check_result('Rshadow_scaled',{'R':r,'Rshadow_scaled':changed})


if __name__=='__main__':unittest.main()
