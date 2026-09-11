import copy
import json
from pathlib import Path,PurePosixPath
import sys
import unittest
import tempfile
import shutil
import importlib.util
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mechanism_candidates import ROOT,BASE,read,read_vector
from prepare_mechanism_screen import select,norm
from run_mechanism_screen import command,validate_arm,validate_bundle
from grade_mechanism_screen import compare


class ScreenTests(unittest.TestCase):
    def test_split_excludes_indices_and_whitespace_duplicates(self):
        train=[dict(problem='Question '+str(i)) for i in range(500)]
        train[4]['problem']='Question\n 0'
        screen,reserve,_=select(train,['Question 1'],{0,2,3})
        self.assertEqual((len(screen),len(reserve)),(100,200))
        self.assertFalse(set(screen)&set(reserve))
        self.assertFalse(set(screen+reserve)&{0,1,2,3,4})
        self.assertEqual(len({norm(train[i]['problem']) for i in screen+reserve}),300)
        self.assertEqual(select(train,['Question 1'],{0,2,3})[0],screen)

    def fixture(self):
        runtime=dict(max_tokens=16000,temperature=.7,async_scheduling=True)
        plan=dict(runtime=runtime,dynamic_parameters={'q25c':.8},model='/model',dataset_sha256='data',
            arms={'original_dynamic':dict(vector_sha256='vec',fit_sha256='fit',directory='assets/original_dynamic')})
        rows=[dict(problem='q'+str(i),answer=str(i),train_index=i+1000) for i in range(100)]
        records=[dict(dataset_index=i,problem=r['problem'],gold=r['answer'],token_ids=[1]*10,tokens=10,
                      thinking_tokens=8,finish_reason='stop') for i,r in enumerate(rows)]
        saved=dict(status='diagnostic_completed',protocol=dict(runtime,offset=0,limit=100,model='/model',
            run_order=['rebalance_dynamic'],easysteer_output_layer=20,dynamic_params={'q25c':.8}),
            provenance=dict(dataset_sha256='data',vector_sha256='vec',calibration_fit_sha256='fit',git_status=''),
            rebalance_dynamic=dict(records=records,summary=dict(generation_seconds=2,grading_errors=0,preemptions=0,dynamic_kv_replay={})))
        return plan,rows,saved

    def test_pair_validation_rejects_missing_mismatched_or_overcap_answers(self):
        plan,rows,saved=self.fixture();validate_arm(saved,plan,rows,'original_dynamic')
        mutations=[lambda s:s['rebalance_dynamic']['records'].pop(),
                   lambda s:s['rebalance_dynamic']['records'][0].update(problem='wrong'),
                   lambda s:s['rebalance_dynamic']['records'][0].update(tokens=16001,token_ids=[1]*16001),
                   lambda s:s['protocol']['dynamic_params'].update(inject_first_step=True),
                   lambda s:s['provenance'].update(vector_sha256='wrong'),
                   lambda s:s['protocol'].update(async_scheduling=False)]
        for mutate in mutations:
            changed=copy.deepcopy(saved);mutate(changed)
            with self.assertRaises(ValueError):validate_arm(changed,plan,rows,'original_dynamic')

    def test_command_never_runs_reserve_or_unsteered_control(self):
        plan,_,_=self.fixture()
        cmd=command(plan,PurePosixPath('/bundle'),PurePosixPath('/out'),'original_dynamic',PurePosixPath('/repo'))
        self.assertIn('/bundle/screen100.jsonl',cmd)
        self.assertNotIn('confirmation200.jsonl',' '.join(cmd))
        self.assertEqual(cmd[cmd.index('--diagnostic-group')+1],'rebalance_dynamic')
        self.assertEqual(cmd[cmd.index('--max-tokens')+1],'16000')
        self.assertIn('--async-scheduling',cmd)

    def test_analysis_keeps_errors_and_caps_and_catches_accuracy_loss(self):
        _,rows,saved=self.fixture();a=saved['rebalance_dynamic'];b=copy.deepcopy(a)
        a['records'][0].update(tokens=16000,thinking_tokens=16000,finish_reason='length')
        b['records'][0].update(tokens=16000,thinking_tokens=16000,finish_reason='length')
        for r in b['records'][1:]:r.update(tokens=5,thinking_tokens=4)
        ga=dict(correct=99,seconds=1,records=[dict(correct=i!=0) for i in range(100)])
        gb=copy.deepcopy(ga);gb['correct']=98;gb['records'][1]['correct']=False
        result=compare(a,b,ga,gb,[r['train_index'] for r in rows])
        self.assertEqual(result['groups']['original_dynamic']['mean_total_tokens'],169.9)
        self.assertEqual(result['groups']['candidate']['capped'],1)
        self.assertEqual(result['degraded_indices'],[1]);self.assertFalse(result['passes_fixed_screen'])
        json.dumps(result,allow_nan=False)

    def test_accuracy_loss_alone_blocks_otherwise_passing_compression(self):
        _,rows,saved=self.fixture();a=saved['rebalance_dynamic'];b=copy.deepcopy(a)
        for r in b['records']:r.update(tokens=5,thinking_tokens=4)
        ga=dict(correct=100,seconds=1,records=[dict(correct=True) for _ in rows])
        gb=copy.deepcopy(ga)
        self.assertTrue(compare(a,b,ga,gb,list(range(100)))['passes_fixed_screen'])
        gb['correct']=99;gb['records'][0]['correct']=False
        self.assertFalse(compare(a,b,ga,gb,list(range(100)))['passes_fixed_screen'])

    def test_prepared_assets_pass_actual_easysteer_payload_schema(self):
        bundle=ROOT/BASE/'configs/mechanism_screen100_20260911'
        plan,_=validate_bundle(bundle)
        path=ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/payloads.py'
        spec=importlib.util.spec_from_file_location('cpu_payload_contract',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original=bundle/'assets/original_dynamic/auto_vector.pt'
        for arm in plan['arms'].values():
            vector=read_vector(bundle/arm['directory']/'auto_vector.pt',original)
            payload=module.DirectionVector({20:vector})
            wire=payload.to_wire()
            np.testing.assert_array_equal(payload.layers[20],vector)
            self.assertEqual(wire['kind'],'direction')

    def test_bundle_tampering_is_rejected(self):
        bundle=ROOT/BASE/'configs/mechanism_screen100_20260911'
        with tempfile.TemporaryDirectory() as directory:
            copy_path=Path(directory)/'bundle';shutil.copytree(bundle,copy_path)
            validate_bundle(copy_path)
            target=copy_path/'assets/min_displacement/auto_vector.pt'
            raw=bytearray(target.read_bytes());raw[-20]^=1;target.write_bytes(raw)
            with self.assertRaises(ValueError):validate_bundle(copy_path)


if __name__=='__main__':unittest.main()
