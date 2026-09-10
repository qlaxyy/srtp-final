import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_cycle_monitor import CycleMonitor, replay, token_bytes
from audit_calibration_labels import ROOT, read, decoder


class CycleMonitorTests(unittest.TestCase):
    def test_alert_needs_second_complete_block_and_is_prefix_causal(self):
        monitor=CycleMonitor()
        self.assertEqual(monitor.feed(b'A\n\nB\n\nA\n\nB',1),[])
        self.assertEqual(monitor.feed(b'\n',2),[])
        event=monitor.feed(b'\n',3)[0]
        snapshot=dict(event)
        self.assertEqual((event['period'],event['detected_token']),(2,3))
        monitor.feed(b'A\n\nB\n\nnew fact\n\n',4)
        self.assertEqual(event,snapshot)
        self.assertEqual(monitor.episodes[0]['completed_copies'],3)
        self.assertEqual(monitor.units[-1]['text'],'new fact')

    def test_numbers_symbols_and_internal_whitespace_are_preserved(self):
        for data in (b'x=1\n\nx=2\n\n',b'x+1\n\nx-1\n\n',b'x =1\n\nx=1\n\n'):
            self.assertEqual(CycleMonitor().feed(data,1),[])
        self.assertEqual(CycleMonitor().feed(b' x=1 \n\nx=1\n\n',1)[0]['period'],1)

    def test_chunked_utf8_and_delimiter_keep_following_math_symbol(self):
        monitor=CycleMonitor()
        self.assertEqual(monitor.feed(b'\xe5',1),[])
        self.assertEqual(monitor.feed(b'\x80\xbc=1\n',2),[])
        self.assertEqual(monitor.feed(b'\n\\[x=1\\]\n\n',3),[])
        self.assertEqual([u['text'] for u in monitor.units],['值=1',r'\[x=1\]'])
        monitor.feed('值=1\n\n\\[x=1\\]\n\n'.encode(),4)
        self.assertEqual(monitor.episodes[0]['period'],2)

    def test_incomplete_tail_and_final_answer_do_not_trigger(self):
        pieces={1:b'Check\n\n',2:b'Check',3:b'</think>',4:b'\n\nCheck\n\n'}
        self.assertFalse(replay([1,2,3,4],pieces,3).episodes)
        self.assertFalse(replay([1,2],pieces,3).episodes)
        self.assertEqual(len(replay([1,1,3,4],pieces,3).episodes),1)

    def test_token_byte_mapping_matches_verified_decoder(self):
        path=ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
        if not path.exists():
            self.skipTest('Optional frozen tokenizer is not stored in Git')
        tokenizer=read(path)
        pieces=token_bytes(tokenizer)
        decode,_=decoder(tokenizer)
        # Cover the actual frozen vocabulary, including split UTF-8 byte tokens.
        for token in pieces:
            self.assertEqual(pieces[token].decode('utf-8',errors='replace'),decode([token]))
        joined=[tokenizer['model']['vocab'][s] for s in ('x','Ċ','Ċ','\\','[')]
        self.assertEqual(b''.join(pieces[t] for t in joined).decode(),decode(joined))


if __name__=='__main__':
    unittest.main()
