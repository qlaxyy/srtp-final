import copy
import json
from pathlib import Path
import tempfile
import unittest

from narrow_grade import decision
from narrow_runner import cases, validate_receipt
from engineering import sha
from prepare_screen import phash


class ProtocolTests(unittest.TestCase):
    def test_accuracy_recovery_can_trade_some_compression_but_not_all(self):
        groups={k:dict(n=100,correct=c,mean_total_tokens=t,mean_thinking_tokens=t-100,capped=1)
                for k,c,t in [('R',90,1000),('RC14',88,800),('RC8',89,850)]}
        self.assertTrue(decision(groups)['promote'])
        groups['RC8']['mean_total_tokens']=1000
        self.assertFalse(decision(groups)['promote'])

    def test_reject_loss_over_margin_more_caps_and_unchanged_candidate(self):
        groups={k:dict(n=100,correct=90,mean_total_tokens=t,mean_thinking_tokens=t-100,capped=1)
                for k,t in [('R',1000),('RC14',900),('RC8',800)]}
        self.assertTrue(decision(groups)['promote'])
        for changes in ({'correct':87},{'capped':2},
                        {'mean_total_tokens':900,'mean_thinking_tokens':800}):
            bad=copy.deepcopy(groups);bad['RC8'].update(changes)
            self.assertFalse(decision(bad)['promote'])

    def test_receipt_rejects_unapproved_or_colliding_data_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'plan.json';audit=root/'audit.json'
            rows=[dict(dataset_index=i,dataset='math_train',problem=str(i),problem_sha256=phash(str(i))) for i in range(100)]
            plan=dict(arms=['R','RC14','RC8'],rows=rows,reserved_rows_sha256='abc',
                runtime=dict(dtype='bfloat16',max_tokens=16000,max_model_len=17408,max_num_seqs=32,
                    max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,
                    chunked_prefill=True,seed=42,temperature=.7,top_p=.95),
                speed_profiles={'current32':{'max_num_seqs':32},'candidate48':{'max_num_seqs':48}})
            p.write_text(json.dumps(plan));audit.write_text(json.dumps(dict(passed=True,rows_sha256='abc')))
            item=dict(path=str(audit),sha256=sha(audit))
            receipt=dict(gpu_authorized=True,plan_sha256=sha(p),data_reconciled=True,
                unregistered_claims_checked=True,local_reconciliation=item,remote_reconciliation=item)
            validate_receipt(plan,receipt,p)
            for key,value in [('gpu_authorized',False),('plan_sha256','bad'),('data_reconciled',False)]:
                bad=dict(receipt,**{key:value})
                with self.assertRaises(AssertionError):validate_receipt(plan,bad,p)
            audit.write_text(json.dumps(dict(passed=False,rows_sha256='abc')))
            item['sha256']=sha(audit)
            with self.assertRaises(AssertionError):validate_receipt(plan,receipt,p)

    def test_engineering_and_effect_arms_are_separate(self):
        self.assertEqual(len(cases({},'engineering')),6)
        self.assertEqual([x[0] for x in cases({},'screen')],['R','RC14','RC8'])
        self.assertEqual(cases({},'speed'),[('R','off','original14')])


if __name__=='__main__':unittest.main(verbosity=2)
