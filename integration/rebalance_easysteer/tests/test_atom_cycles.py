import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_atom_cycles import AtomCycleMonitor, replay_atoms


class AtomCycleTests(unittest.TestCase):
    def test_no_alarm_until_complete_second_copy_and_history_is_immutable(self):
        m=AtomCycleMonitor()
        self.assertFalse(m.feed(b'We check. We che',1))
        self.assertFalse(m.feed(b'ck',2))
        event=m.feed(b'.',3)[0]
        snapshot=dict(event)
        self.assertEqual((event['period_atoms'],event['detected_token']),(3,3))
        m.feed(b' We check. New result. ',4)
        self.assertEqual(event,snapshot)
        self.assertEqual(m.episodes[0]['completed_copies'],3)
        self.assertEqual(m.episodes[0]['closed_on_mismatch_token'],4)

    def test_complete_numbers_variables_and_operators_are_preserved(self):
        for text in ['x=123 x=124 ', 'x1=4 x2=4 ', 'x+2 x-2 ', 'v=1.23 v=1.24 ', 'v=1,234 v=1,235 ']:
            self.assertFalse(AtomCycleMonitor().feed(text.encode(),1),text)
        m=AtomCycleMonitor()
        m.feed(b'value=123 value=12',1)
        self.assertFalse(m.episodes)
        self.assertFalse(m.feed(b'4 ',2))

    def test_numeric_separator_requires_observed_lookahead(self):
        m=AtomCycleMonitor()
        m.feed(b'1,234.50',1)
        self.assertEqual(m.atoms,[])
        m.feed(b'.',2)
        self.assertEqual(m.atoms,[])
        m.feed(b' ',3)
        self.assertEqual([a['text'] for a in m.atoms],['1,234.50','.'])
        self.assertEqual([a['closed_token'] for a in m.atoms],[3,3])

    def test_intrapagraph_formula_and_single_newline_list_are_detected(self):
        for text in [b'(x+1)=(x+1)=(x+1)=',b'- x=1\n- y=2\n- x=1\n- y=2\n']:
            m=AtomCycleMonitor()
            m.feed(text,1)
            self.assertTrue(m.episodes,text)

    def test_whitespace_only_is_ignored_and_single_atom_runs_excluded(self):
        m=AtomCycleMonitor()
        m.feed(b'x = 1\n x=1 ',1)
        self.assertTrue(m.episodes)
        for text in [b'0 0 0 0 ',b'yes yes yes ',b'---',b'000000000 ',b'banana ',b'x=1 X=1 ']:
            self.assertFalse(AtomCycleMonitor().feed(text,1),text)

    def test_split_utf8_and_pending_tail_have_exact_spans(self):
        text='值=1 值=1 '
        m=AtomCycleMonitor()
        data=text.encode()
        for n, byte in enumerate(data,1):
            m.feed(bytes([byte]),n)
        self.assertTrue(m.episodes)
        for atom in m.atoms:
            self.assertEqual(data[atom['start_byte']:atom['end_byte']].decode(),atom['text'])
        m.feed(b'\xe5',len(data)+1)
        self.assertEqual(m.decoder.getstate()[0],b'\xe5')

    def test_eof_and_final_answer_cannot_complete_a_pending_cycle(self):
        pieces={1:b'x=1 x=1',2:b'</think>',3:b' x=1 ',4:b' '}
        self.assertFalse(replay_atoms([1],pieces,2).episodes)
        self.assertFalse(replay_atoms([1,2,3],pieces,2).episodes)
        self.assertTrue(replay_atoms([1,4,2,3],pieces,2).episodes)

    def test_first_alert_matches_independent_brute_force_suffix_check(self):
        rng=random.Random(42)
        for _ in range(80):
            atoms=[rng.choice(['aa','bb','cc','12','13','+','-']) for _ in range(70)]
            expected=None
            for n in range(1,len(atoms)+1):
                periods=[p for p in range(2,n//2+1) if len(set(atoms[n-p:n]))>=2 and atoms[n-2*p:n-p]==atoms[n-p:n]]
                if periods:
                    expected=(n,min(periods))
                    break
            m=AtomCycleMonitor()
            for n, atom in enumerate(atoms,1):
                m.feed((atom+' ').encode(),n)
            actual=(m.episodes[0]['detected_end_atom'],m.episodes[0]['period_atoms']) if m.episodes else None
            self.assertEqual(actual,expected)


if __name__=='__main__':
    unittest.main()
