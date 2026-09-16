import unittest
import numpy as np
from repeatability import ARMS,SEEDS,schedule,ordered_rows,point_decision
from repeatability_grade import compare

class RepeatabilityTests(unittest.TestCase):
    def test_balanced_arm_order_and_shared_seed_order(self):
        s=schedule();self.assertEqual(len(s),9)
        for arm in ARMS:self.assertEqual({i%3 for i,x in enumerate(s) if x['arm']==arm},{0,1,2})
        rows=[dict(problem_sha256=str(i),dataset_index=i) for i in range(100)]
        for seed in SEEDS:
            a=ordered_rows(rows,seed);self.assertEqual(a,ordered_rows(list(reversed(rows)),seed))
            self.assertEqual(sorted(r['dataset_index'] for r in a),list(range(100)))

    def test_identical_candidate_never_promotes(self):
        a=np.ones((3,5,4));a[...,1:3]=100
        c=compare(a,a,bootstrap=30)
        self.assertEqual(c['question_cluster_ci95']['total_change_percent'],[0,0])
        self.assertFalse(c['interval_support'])
        self.assertFalse(point_decision(dict(RChistory_vs_RC14=c,RChistory_vs_R=c))['point_promising'])

    def test_clustered_comparison_and_paired_transitions(self):
        a=np.ones((3,5,4));a[...,1:3]=100;a[...,3]=0;a[:,0,0]=0
        b=a.copy();b[...,1:3]=90;b[:,0,0]=1;b[:,1,0]=0
        c=compare(a,b,bootstrap=30)
        self.assertAlmostEqual(c['total_change_percent'],-10)
        self.assertEqual(c['per_seed'][0]['wrong_to_right'],[0]);self.assertEqual(c['per_seed'][0]['right_to_wrong'],[1])
        self.assertTrue(point_decision(dict(RChistory_vs_RC14=c,RChistory_vs_R=c))['point_promising'])
        c['cap_delta']=1
        self.assertFalse(point_decision(dict(RChistory_vs_RC14=c,RChistory_vs_R=c))['point_promising'])

if __name__=='__main__':unittest.main(verbosity=2)
