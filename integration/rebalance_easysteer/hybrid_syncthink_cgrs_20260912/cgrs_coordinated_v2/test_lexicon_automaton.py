import unittest
from pathlib import Path
import numpy as np
from prepare_label_alignment import load
from lexicon_automaton import Automaton,expanded_terms
from large_lexicon_reference import OpeningMatcher

ROOT=Path(__file__).resolve().parents[4]
AUTHOR=load('automaton_test_author',ROOT.parent/'srtp-final/sources/ReBalance/hidden_analysis_auto.py')


class AutomatonTests(unittest.TestCase):
    def test_expansion(self):
        terms=expanded_terms(AUTHOR.LEXICON_BASE)
        for term in terms:self.assertTrue(AUTHOR.has_lexicon_hit(term),term)

    def test_word_end_is_delayed(self):
        a=Automaton(expanded_terms(AUTHOR.LEXICON_BASE))
        state,hit=a.consume(0,'Wait')
        self.assertFalse(hit);self.assertTrue(a.final[state])
        state,hit=a.consume(state,'ForSeconds')
        self.assertFalse(hit);self.assertFalse(a.final[state])

    def test_search_and_arbitrary_segmentation(self):
        a=Automaton(expanded_terms(AUTHOR.LEXICON_BASE))
        cases=['WaitForSeconds','I was correct','differently','differentials',
            'let me check','let---me\ncheck','Think again.','However, yes',
            'xwait','wait_x','wait2','Wait,','Hmm','alternative','alternatively',
            'something about whether this holds','Carefully','confused','sometimes',
            'waİt','_wait','anotherK','but—perhaps','foo alternative bar']
        for text in cases:
            expected=bool(AUTHOR.has_lexicon_hit(text))
            for split in range(len(text)+1):
                s,h=a.consume(0,text[:split]);s,h2=a.consume(s,text[split:])
                self.assertEqual(h or h2 or bool(a.final[s]),expected,(text,split))

    def test_opening_completion_oracle(self):
        a=Automaton(expanded_terms(AUTHOR.LEXICON_BASE),opening=True)
        for prefix in ['', ' ', 'Let','Let me','Let me ', 'I','I was', 'We should',
                       'Alter','Perhaps','  hold-', 'Think ', 'butter']:
            for piece in [' check',' checkmate','t','natively',' on',' again','Wait',', yes',' Hmm',' ']:
                oracle=OpeningMatcher(AUTHOR);oracle.accept('<think>',start=True)
                oracle.accept('\n\n',boundary=True)
                # Prefix may be partial but must not already have completed a word.
                s,h=a.consume(0,prefix)
                if h or a.final[s]:continue
                oracle.accept(prefix)
                ns,nh=a.consume(s,piece)
                self.assertEqual(nh or bool(a.final[ns]),oracle.would_complete(piece),(prefix,piece))

    def test_token_table_matches_character_reference(self):
        a=Automaton(expanded_terms(AUTHOR.LEXICON_BASE),opening=True)
        pieces=['','Wait',' wait','Let',' me',' check','For','\n\n','-','However, yes']
        tr,hit=a.compile_tokens(pieces)
        for s in range(len(a.states)):
            for i,piece in enumerate(pieces):
                ns,h=a.consume(s,piece)
                self.assertEqual(int(tr[s,i]),ns)
                self.assertEqual(bool(hit[s,i]),h or bool(a.final[ns]))


if __name__=='__main__':unittest.main()
