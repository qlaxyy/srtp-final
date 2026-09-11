import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import test_repeat_validation as fixtures
from run_question_balanced_validation import arm_plan,combine,analyze,old_command


class PairTests(unittest.TestCase):
    def fixture(self):
        rows,old,a,_=fixtures.ValidationTest().fixture()
        candidate=dict(assets='new_assets',vector_sha256='new_vector',fit_sha256='new_fit',
                       dynamic_parameters={'initial_coef':-1,'low_val_2':-1.4})
        plan=dict(old,run_order=['original_dynamic','question_balanced'],
            calibrations={'original_dynamic':{k:old[k] for k in candidate},'question_balanced':candidate})
        b=copy.deepcopy(a)
        b['protocol']['dynamic_params']=candidate['dynamic_parameters']
        b['provenance'].update(vector_sha256='new_vector',calibration_fit_sha256='new_fit')
        return rows,plan,a,b

    def test_commands_differ_only_in_calibration_and_output(self):
        _,p,_,_=self.fixture()
        a,b=[old_command(arm_plan(p,arm),Path('data'),Path(arm+'.json'),'off') for arm in p['run_order']]
        differences=[i for i,(x,y) in enumerate(zip(a,b,strict=True)) if x!=y]
        self.assertEqual(differences,[a.index('--vector')+1,a.index('--calibration-fit')+1,a.index('--output')+1])

    def test_rejects_old_vector_in_candidate_and_keeps_all_records(self):
        rows,p,a,b=self.fixture()
        pair=combine(a,b,p,rows)
        grades=dict(groups={k:dict(records=[{}]*100,author_correct=80) for k in ('baseline','rebalance_dynamic')},
                    improved_indices=[],degraded_indices=[])
        result=analyze(pair,grades)
        self.assertEqual(result['groups']['question_balanced']['capped'],1)
        self.assertEqual(result['groups']['question_balanced']['mean_total_tokens'],161.98)
        for mutation in ('vector','fit','parameters','mode','missing'):
            bad=copy.deepcopy(b)
            if mutation=='vector': bad['provenance']['vector_sha256']='vector'
            elif mutation=='fit': bad['provenance']['calibration_fit_sha256']='fit'
            elif mutation=='parameters': bad['protocol']['dynamic_params']['low_val_2']=-1.5
            elif mutation=='mode': bad['protocol']['repeat_gate']={'mode':'cancel_positive'}
            else: bad['rebalance_dynamic']['records'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): combine(a,bad,p,rows)


if __name__=='__main__': unittest.main()
