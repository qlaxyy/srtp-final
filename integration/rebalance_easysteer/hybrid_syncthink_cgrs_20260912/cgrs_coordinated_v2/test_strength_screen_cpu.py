import copy
import unittest
from engineering import HERE,read
from narrow_runner import cases,validate_receipt,validate_engineering_gate
from strength_screen import validate,decision,ARMS
from strength_statistics import comparisons


class StrengthScreenTests(unittest.TestCase):
    def setUp(self):
        self.folder=HERE/'strength_screen100_20260917';self.plan=read(self.folder/'plan.json')
    def test_fixed_scope_and_engineering_proof(self):
        validate(self.plan);self.assertEqual([r[0] for r in cases(self.plan,'screen')],ARMS)
        validate_engineering_gate(self.plan,self.folder/'plan.json',self.folder/'engineering_gate.json')
        for phase in ('engineering','full','speed'):
            with self.assertRaises(AssertionError):cases(self.plan,phase)
    def test_unapproved_or_changed_scope_rejected(self):
        with self.assertRaises(AssertionError):validate_receipt(self.plan,read(self.folder/'receipt_template.json'),self.folder/'plan.json')
        for key,value in [('seed',142),('max_tokens',512),('max_num_seqs',256)]:
            p=copy.deepcopy(self.plan);p['runtime'][key]=value
            with self.assertRaises(AssertionError):validate(p)
    def test_screen_rule_does_not_reward_accuracy_only_or_equal_tokens(self):
        g={a:dict(n=100,correct=80,mean_total_tokens=1000,mean_thinking_tokens=900,capped=2) for a in ARMS}
        g['RCscaled']['correct']=85;self.assertFalse(decision(g)['retain_for_separate_confirmation'])
        g['RCscaled'].update(correct=78,mean_total_tokens=999,mean_thinking_tokens=899)
        self.assertTrue(decision(g)['retain_for_separate_confirmation'])
        g['RCscaled']['correct']=77;self.assertFalse(decision(g)['retain_for_separate_confirmation'])
        g['RCscaled']['correct']=80;g['RCscaled']['capped']=3
        self.assertFalse(decision(g)['retain_for_separate_confirmation'])
    def test_paired_resampling_exact_scaling_and_flips(self):
        rows=[dict(dataset_index=i,train_index=i+100,problem_sha256=str(i),correct=bool(i%2),tokens=100+i,thinking_tokens=80+i,finish_reason='stop') for i in range(8)]
        raw={a:copy.deepcopy(rows) for a in ARMS}
        for r in raw['RCscaled']:r['tokens']*=.5;r['thinking_tokens']*=.5
        result=comparisons(raw,draws=100)
        c=result['RCscaled_vs_R'];self.assertEqual(c['paired_ci95']['total_change_percent'],[-50.,-50.])
        self.assertEqual(c['paired_ci95']['accuracy_delta_pp'],[0.,0.])
        raw['RCscaled'][0]['correct']=True;raw['RCscaled'][1]['correct']=False
        c=comparisons(raw,draws=100)['RCscaled_vs_R']
        self.assertEqual(c['wrong_to_correct'][0]['train_index'],100)
        self.assertEqual(c['correct_to_wrong'][0]['train_index'],101)
        raw['RCscaled'].reverse()
        with self.assertRaises(AssertionError):comparisons(raw,draws=100)


if __name__=='__main__':unittest.main()
