import copy,unittest
from engineering import HERE,read
from history_full import validate_full_plan
from narrow_runner import cases


class FullTests(unittest.TestCase):
    def test_exact_datasets_runtime_and_single_candidate(self):
        old=read(HERE/'full_7b_run1/plan.json')
        plan=dict(candidate_kind='first_reflection_full',phase='full_test',arms=['RChistory'],
            datasets=old['datasets'],runtime=dict(old['runtime'],max_model_len=17920))
        receipt=dict(authorized_phases=['full'],exposed_test_reuse_acknowledged=True)
        validate_full_plan(plan,receipt)
        self.assertEqual([x[0] for x in cases(plan,'full')],['math_test','gsm8k_test'])
        self.assertTrue(all(x[3]=='after_first_reflection' for x in cases(plan,'full')))
        for phase in ('screen','engineering','speed'):
            with self.assertRaises(AssertionError):cases(plan,phase)
        for mutate in ('rows','runtime','receipt'):
            p=copy.deepcopy(plan);r=dict(receipt)
            if mutate=='rows':p['datasets']['math_test']['rows'].pop()
            elif mutate=='runtime':p['runtime']['max_num_seqs']=48
            else:r['exposed_test_reuse_acknowledged']=False
            with self.assertRaises(AssertionError):validate_full_plan(p,r)


if __name__=='__main__':unittest.main(verbosity=2)
