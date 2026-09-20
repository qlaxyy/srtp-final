import unittest
from audit_first_answer import compare, first_answer


class FirstAnswerTests(unittest.TestCase):
    def test_conservative_comparison(self):
        self.assertEqual(compare('0.5', r'\frac{1}{2}'), 'equal_rational')
        self.assertEqual(compare('2', r'\frac{1}{3}'), 'different_rational')
        self.assertEqual(compare('x+x', '2x'), 'unknown')
        self.assertEqual(compare('1', r'\frac{1}{0}'), 'unknown')

    def test_prefix_and_answer_phase(self):
        prefix = r'thus \boxed{\frac{1}{2}}'
        self.assertEqual(first_answer(prefix)[0], first_answer(prefix+r' later \boxed{2}')[0])
        self.assertIsNone(first_answer(r'work </think> \boxed{2}')[0])
        self.assertIsNone(first_answer(r'work \boxed{2')[0])


if __name__ == '__main__':
    unittest.main()
