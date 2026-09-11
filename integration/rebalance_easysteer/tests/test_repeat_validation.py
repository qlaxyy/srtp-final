"""CPU checks for independent selection and rejecting invalid paired results."""
import copy
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_repeat_validation import select
from run_repeat_validation import command, combine, analyze, validate_arm, main


class ValidationTest(unittest.TestCase):
    def test_default_preview_never_launches_a_subprocess(self):
        bundle = Path(__file__).resolve().parents[1]/'configs/repeat_validation100_20260911'
        argv = ['runner','--bundle',str(bundle),'--output','unused_preview_output']
        with patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()) as captured:
            with patch('run_repeat_validation.subprocess.run',side_effect=AssertionError('Unexpected process')):
                with patch('run_repeat_validation.subprocess.check_output',side_effect=AssertionError('Unexpected import or GPU process')):
                    main()
        self.assertIn('cpu_preflight_passed_no_generation',captured.getvalue())
        self.assertFalse(Path('unused_preview_output').exists())

    def fixture(self):
        rows = [dict(problem=f'p{i}', answer=str(i)) for i in range(100)]
        plan = dict(runtime=dict(max_tokens=16000, async_scheduling=True),
                    model='model', decoder_output_layer=20, dynamic_parameters={'initial_coef':-1},
                    dataset_sha256='data',vector_sha256='vector',fit_sha256='fit',assets='assets')
        protocol = dict(plan['runtime'], offset=0,limit=100,model='model',
                        easysteer_output_layer=20,run_order=['rebalance_dynamic'],
                        dynamic_params=plan['dynamic_parameters'])
        records = [dict(dataset_index=i, problem=r['problem'], gold=r['answer'],
                        tokens=2,token_ids=[1,2],thinking_tokens=1,finish_reason='stop') for i,r in enumerate(rows)]
        records[-1].update(tokens=16000,token_ids=[1]*16000,thinking_tokens=16000,finish_reason='length')
        summary = dict(generation_seconds=10,grading_errors=0,preemptions=0,dynamic_kv_replay={'suspended':0,'restored':0})
        arm = dict(status='diagnostic_completed',protocol=protocol,
                   provenance=dict(dataset_sha256='data',vector_sha256='vector',calibration_fit_sha256='fit',commit='code',git_status=''),
                   rebalance_dynamic=dict(records=records,summary=summary),environment={},calibration={})
        candidate = copy.deepcopy(arm)
        candidate['protocol']['repeat_gate'] = dict(mode='cancel_positive')
        candidate['rebalance_dynamic']['summary']['repeat_gate'] = dict(changed_scales=2)
        return rows, plan, arm, candidate

    def test_selection_excludes_indices_and_whitespace_duplicates(self):
        train = [dict(problem=f'problem{i}') for i in range(750)]
        train[710]['problem'] = ' p r o b l e m 0 '
        train[711]['problem'] = 'p r o b l e m 11'
        train[712]['problem'] = 't e s t'
        args = (train,[dict(problem='test')],range(10),range(10,20))
        chosen, count = select(*args)
        self.assertEqual((chosen,count),select(*args))
        self.assertEqual(len(set(chosen)),100)
        self.assertFalse(set(chosen) & (set(range(20))|{710,711,712}))
        self.assertEqual(count,727)

    def test_commands_change_only_output_and_gate(self):
        _, plan, _, _ = self.fixture()
        a = command(plan,Path('data'),Path('a.json'),'off')
        b = command(plan,Path('data'),Path('b.json'),'cancel_positive')
        differences = [(x,y) for x,y in zip(a,b,strict=True) if x != y]
        self.assertEqual(differences,[('off','cancel_positive'),('a.json','b.json')])
        self.assertIn('--async-scheduling',a)
        self.assertEqual(a[a.index('--diagnostic-group')+1],'rebalance_dynamic')

    def test_pair_rejects_missing_records_wrong_mode_and_sampling(self):
        rows, plan, a, b = self.fixture()
        pair = combine(a,b,plan,rows)
        self.assertEqual(len(pair['baseline']['records']),100)
        for mutation in ('missing','mode','sampling','prompt','first_step','wrong_baseline'):
            bad = copy.deepcopy(b)
            if mutation == 'missing': bad['rebalance_dynamic']['records'].pop()
            elif mutation == 'mode': bad['protocol']['repeat_gate']['mode']='shadow'
            elif mutation == 'sampling': bad['protocol']['max_tokens']=4000
            elif mutation == 'prompt': bad['rebalance_dynamic']['records'][9]['problem']='other'
            elif mutation == 'first_step': bad['protocol']['dynamic_params']['inject_first_step']=True
            else: bad['baseline']={}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_arm(bad,plan,rows,'cancel_positive')

    def test_summary_keeps_capped_and_wrong_answers(self):
        rows, plan, a, b = self.fixture()
        pair = combine(a,b,plan,rows)
        grades = dict(groups={name:dict(records=[{}]*100,author_correct=80) for name in ('baseline','rebalance_dynamic')},
                      improved_indices=[],degraded_indices=[])
        report = analyze(pair,grades)
        self.assertEqual(report['groups']['original_dynamic']['mean_total_tokens'],161.98)
        self.assertEqual(report['groups']['repeat_cancel_positive']['capped'],1)
        self.assertEqual(report['total_token_change_percent'],0)
        self.assertEqual(report['decision'],'Do not expand or retune this validation set.')


if __name__ == '__main__':
    unittest.main()
