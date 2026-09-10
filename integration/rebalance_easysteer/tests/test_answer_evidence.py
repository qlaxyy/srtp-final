"""Protect evidence provenance and avoid converting lookup failures into labels."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_answer_evidence import compare_literals, prefix_evidence, validate_anchor, save


class AnswerEvidenceTests(unittest.TestCase):
    def test_exact_equivalence_without_rounding(self):
        self.assertEqual(compare_literals('54/210', r'\frac{9}{35}'), 'correct')
        self.assertEqual(compare_literals('0.16', '4/25'), 'correct')
        self.assertEqual(compare_literals('-2.5', r'-\frac{5}{2}'), 'correct')
        self.assertEqual(compare_literals('0.04', '0.16'), 'incorrect')

    def test_unparsed_text_is_not_an_incorrect_answer(self):
        for value in ['water 0.16 liters', '0.16 or 0.04', '1/0', r'\sqrt{3}', '(3,4)']:
            with self.subTest(value=value):
                self.assertEqual(compare_literals(value, '0.16'), 'unverified')

    def test_no_future_answer_leaks_to_earlier_prefix(self):
        anchors = [dict(step=8, scope='original_problem', verdict='correct'),
                   dict(step=9, scope='original_problem', verdict='incorrect')]
        self.assertEqual(prefix_evidence(anchors, 7)['answer_evidence'], 'no_reviewed_answer_evidence')
        self.assertEqual(prefix_evidence(anchors, 8)['answer_evidence'], 'correct_claim_observed')
        after = prefix_evidence(anchors, 9)
        self.assertEqual(after['answer_evidence'], 'conflicting_observed_claims')
        self.assertEqual(after['latest_reviewed_answer_verdict'], 'incorrect')

    def test_intermediate_and_rejected_values_do_not_become_answers(self):
        anchors = [dict(step=5, scope='intermediate', verdict='correct'),
                   dict(step=6, scope='retracted_or_hypothetical', verdict='incorrect')]
        self.assertEqual(prefix_evidence(anchors, 10)['reviewed_answer_steps'], [])

    def test_boundary_punctuation_and_bad_source_positions(self):
        question = dict(gold_answer='(3,4)', steps=[dict(step=1, start=0, stop=5,
                        evidence_stop=6, evidence_text='range (3,4)\n\n')])
        anchor = dict(step=1, quote='(3,4)', answer='(3,4)', scope='original_problem',
                      verdict='correct', basis='manual interval check')
        self.assertEqual(validate_anchor(anchor, question)['evidence_stop'], 6)
        for replacement in [dict(step=0), dict(quote='(3,5)'), dict(context_steps=[2])]:
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                validate_anchor(dict(anchor, **replacement), question)

    def test_scalar_verdict_must_match_exact_check(self):
        question = dict(question='fixture', gold_answer='0.16', steps=[dict(step=1, start=0, stop=5,
                        evidence_stop=6, evidence_text='water 0.04 liters')])
        anchor = dict(step=1, quote='water 0.04 liters', answer='0.04', scope='original_problem',
                      verdict='correct', basis='deliberately inconsistent annotation')
        with self.assertRaises(AssertionError):
            validate_anchor(anchor, question)

    def test_existing_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            save(path, {'old': True})
            with self.assertRaises(FileExistsError):
                save(path, {'replacement': True})
            self.assertIn('old', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
