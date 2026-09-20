"""CPU semantic checks; these do not validate a native GPU adapter."""
import unittest
from pathlib import Path
import numpy as np
import torch
from prepare_label_alignment import labels, lexical_coefficient, load
from large_lexicon_reference import OpeningMatcher


class LabelTests(unittest.TestCase):
    def matcher(self):
        root = Path(__file__).resolve().parents[4]
        author = load('test_alignment_author', root.parent/'srtp-final/sources/ReBalance/hidden_analysis_auto.py')
        matcher = OpeningMatcher(author)
        matcher.accept('<think>', start=True)
        matcher.accept('\n\n', boundary=True)
        return matcher

    def test_phrase_completion_not_first_word(self):
        matcher = self.matcher()
        self.assertFalse(matcher.would_complete('Let'))
        matcher.accept('Let')
        self.assertFalse(matcher.would_complete(' me'))
        matcher.accept(' me')
        self.assertTrue(matcher.would_complete(' check'))
        self.assertFalse(matcher.would_complete(' calculate'))
        matcher.accept(' check')
        self.assertFalse(matcher.would_complete(' again'))

    def test_boundaries_and_reset(self):
        matcher = self.matcher()
        self.assertTrue(matcher.would_complete(' Wait'))
        self.assertFalse(matcher.would_complete(' WaitForSeconds'))
        self.assertFalse(matcher.would_complete(' Hmm'))
        matcher.accept('\n\nTherefore', boundary=True)
        self.assertFalse(matcher.would_complete(' Wait'))
        matcher.reset()
        self.assertFalse(matcher.would_complete(' Wait'))

    def test_lexical_precedes_high_confidence(self):
        p = dict(q25c=.2, q75c=.8, q25v=.01, q75v=.04)
        c = np.array([.9, .9, .1, .5, .8, .2])
        v = np.zeros(6)
        lex = np.array([True, False, False, False, False, False])
        over, under = labels(c, v, lex, p, 'lexical')
        self.assertEqual(over.tolist(), [True, False, True, False, False, False])
        self.assertEqual(under.tolist(), [False, True, False, False, False, False])

    def test_joint_requires_both_axes_and_includes_endpoints(self):
        p = dict(q25c=.2, q75c=.8, q25v=.01, q75v=.04)
        c = np.array([.2, .2, .8, .8, .5])
        v = np.array([.04, 0, .01, .05, .1])
        over, under = labels(c, v, np.ones(5, dtype=bool), p, 'confidence_variance')
        self.assertEqual(over.tolist(), [True, False, False, False, False])
        self.assertEqual(under.tolist(), [False, False, True, False, False])

    def test_lexical_reference_uses_exact_labels_and_finite_middle(self):
        root = Path(__file__).resolve().parents[4]
        # Load frozen main controller, not a copied implementation.
        path = root.parent/'srtp-final/sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
        runtime = load('test_alignment_runtime', path)
        p = runtime.ReBalanceParams(boundary_token_ids=(1,), think_start_token_id=2,
            think_end_token_id=3, q25c=.2, q75c=.8, low_val_1=-.5,
            low_val_2=-1.5, high_val_2=.1, curve_tau=.01)
        c = torch.tensor([.9, .9, .1, .5], dtype=torch.float64)
        result = lexical_coefficient(c, torch.tensor([True, False, False, False]), p, runtime)
        self.assertEqual(result[:3].tolist(), [-1.5, .1, -1.5])
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(result[3], 0)


if __name__ == '__main__':
    unittest.main()
