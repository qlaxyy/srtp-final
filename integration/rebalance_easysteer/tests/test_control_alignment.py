"""Checks for time alignment and the CPU-only controller projection."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_control_alignment as audit


class ControlAlignmentTests(unittest.TestCase):
    def step(self, n, value, boundary=True):
        row = {k: 0. for k in ['confidence', 'variance', 'coefficient_float32_emulation',
            'base', 'low_gate_weight', 'high_gate_weight', 'low_adjustment', 'high_adjustment']}
        return dict(**row, step=n, coefficient=value, has_boundary=boundary, boundary_token_offset=n*10)

    def point(self, n):
        return dict(question='Q01', train_index=0, case_id='case', step=n,
            progress_evidence='new', math_validity='supported', progress_quote='evidence',
            answer_evidence='no_reviewed_answer_evidence', latest_reviewed_answer_step=None,
            latest_reviewed_answer_verdict=None)

    def test_incoming_uses_previous_step_not_current_or_future(self):
        steps = [self.step(1, -.8), self.step(2, .1), self.step(3, -1.5)]
        point = audit.join_point(self.point(2), steps)
        self.assertEqual(point['incoming']['coefficient'], -.8)
        self.assertEqual(point['outgoing']['coefficient'], .1)

    def test_no_invented_first_prompt_or_think_end_injection(self):
        p = audit.join_point(self.point(1), [self.step(1, -.8, boundary=False)])
        self.assertIsNone(p['incoming']['coefficient'])
        self.assertIsNone(p['outgoing']['coefficient'])
        self.assertIn('think_end', p['outgoing']['status'])

    def test_saved_base_curve_anchors_and_coefficient_bounds(self):
        params = json.loads((audit.CONFIG / 'auto_code_v2_1p5b_20260908.json').read_text())['parameters']
        p, runtime = SimpleNamespace(**params), audit.load_controller()
        z = runtime['_curve_constants'](p.q25c, p.q75c, p.low_val_1, 'cpu', np.dtype('float64'), p.curve_tau)
        anchors = runtime['_baseline'](audit.tensor([p.q25c, p.q75c, 1]), *z[:4])
        np.testing.assert_allclose(anchors, [p.low_val_1, 0., p.curve_tau], atol=1e-12, rtol=0)
        c, v = audit.tensor([0., 1.]), audit.tensor([.25, 0.])
        actual = runtime['compute_rebalance_coefficient'](c, v, p)
        self.assertAlmostEqual(actual[0], p.low_val_2, places=12)
        self.assertAlmostEqual(actual[1], p.high_val_2, places=12)

    def test_no_boundary_is_not_counted_as_neutral(self):
        rows = [audit.join_point(self.point(1), [self.step(1, -.8)])]
        summary = audit.summarize(rows, 'incoming', -.5)['new_supported']
        self.assertEqual(summary['excluded'], 1)
        self.assertEqual(summary['signs'], {})

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_text('keep')
            with self.assertRaises(FileExistsError):
                audit.run(Path(folder), path)
            self.assertEqual(path.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
