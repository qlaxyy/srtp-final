import unittest
from audit_reasoning_tradeoffs import boxes,repeated_box_prefix,first_difference,summarize

class AuditTests(unittest.TestCase):
    def test_prefix_difference_including_termination(self):
        self.assertIsNone(first_difference([1,2],[1,2]))
        self.assertEqual(first_difference([1,2],[1,3]),1)
        self.assertEqual(first_difference([1],[1,2]),1)

    def test_nested_incomplete_and_escaped_braces(self):
        self.assertEqual(boxes(r'\boxed{\frac{1}{2}}')[0]['value'],r'\frac{1}{2}')
        self.assertEqual(boxes(r'\boxed{\frac{1}{2}'),[])
        self.assertEqual(boxes(r'\boxed{\{a\}}')[0]['value'],r'\{a\}')

    def test_answer_phase_never_activates(self):
        self.assertIsNone(repeated_box_prefix(r'\boxed{2}</think>\boxed{2}'))
        self.assertIsNone(repeated_box_prefix(r'\boxed{2} \boxed{3} \boxed{2}'))
        self.assertEqual(repeated_box_prefix(r'\boxed{ 2 } \boxed{2} trailing')['remaining_thinking_chars'],9)

    def test_future_output_does_not_change_first_detection(self):
        s=r'\boxed{2} then \boxed{2}'
        self.assertEqual(repeated_box_prefix(s)['offset'],repeated_box_prefix(s+r' \boxed{3}')['offset'])

if __name__=='__main__':unittest.main(verbosity=2)
