"""CPU-only, prefix-causal exact repeated-block monitor for saved reasoning.

Alerts are text witnesses, not correct-answer/overthinking/stop decisions.
The CLI never loads a model or imports the generation engine.
"""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tarfile
import time

from audit_calibration_labels import ROOT, read, save, sha, decoder


class CycleMonitor:
    """Feed only arrived token bytes. Exposed event fields use prefix data."""

    def __init__(self):
        self.buffer = bytearray()
        self.buffer_start = 0
        self.units = []
        self.symbols = []
        self.intern = {}
        self.episodes = []
        self.active = None

    def feed(self, piece, token_end):
        self.buffer.extend(piece)
        emitted = []
        while (offset := self.buffer.find(b'\n\n')) >= 0:
            text = bytes(self.buffer[:offset]).strip()
            start = self.buffer_start
            stop = start + offset
            del self.buffer[:offset + 2]
            self.buffer_start = stop + 2
            if not text:
                continue
            unit = dict(text=text.decode('utf-8', errors='replace'),
                        normalized_bytes_sha256=hashlib.sha256(text).hexdigest(), start_byte=start,
                        end_byte=stop, delimiter_end_byte=stop + 2,
                        token_end=token_end)
            self.units.append(unit)
            self.symbols.append(self.intern.setdefault(text, len(self.intern)))
            n = len(self.symbols)
            if self.active is not None:
                period = self.active['period']
                if self.symbols[-1] == self.symbols[-1-period]:
                    self.active['observed_end_unit'] = n
                    self.active['completed_copies'] = (n-self.active['start_unit'])//period
                    continue
                self.active = None
            for period in range(1, n//2 + 1):
                if self.symbols[-1] != self.symbols[-1-period]:
                    continue
                if self.symbols[n-2*period:n-period] == self.symbols[n-period:n]:
                    event = dict(start_unit=n-2*period, period=period,
                                 detected_end_unit=n, detected_token=token_end,
                                 detected_end_byte=stop+2)
                    self.active = dict(**event, observed_end_unit=n, completed_copies=2)
                    self.episodes.append(self.active)
                    # Copy the event so subsequent growth cannot mutate history.
                    emitted.append(event)
                    break
        return emitted


def token_bytes(tokenizer):
    assert tokenizer['decoder']['type'] == 'ByteLevel'
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    inverse = {chr(c): b for c, b in zip(chars, values)}
    raw = {i: bytes(inverse[c] for c in token) for token, i in tokenizer['model']['vocab'].items()}
    for item in tokenizer['added_tokens']:
        raw[item['id']] = item['content'].encode('utf-8')
    return raw


def replay(ids, pieces, think_end):
    monitor = CycleMonitor()
    for n, token in enumerate(ids, 1):
        if token == think_end:
            break
        monitor.feed(pieces[token], n)
    return monitor


def episode_record(episode, monitor, byte_ends):
    start, period = episode['start_unit'], episode['period']
    block = monitor.units[start:start+period]
    first_byte = block[0]['start_byte']
    last_byte = block[-1]['delimiter_end_byte']
    token_start = bisect_left(byte_ends, first_byte + 1)
    token_stop = bisect_left(byte_ends, last_byte) + 1
    complete_end = start + episode['completed_copies']*period
    return dict(**episode, block=block, block_token_span=[token_start, token_stop],
                block_utf8_bytes=last_byte-first_byte,
                complete_repeat_end_token=monitor.units[complete_end-1]['token_end'],
                note='Block token span encloses text and delimiter; a boundary token may contain adjacent text.')


def run(output, plan_path):
    plan = read(plan_path)
    # Reserve a new output directory before doing any work; failed attempts keep
    # their protocol and may not be silently reused as completed runs.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save(output/'plan.json', plan)
    tokenizer_path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    assert sha(tokenizer_path) == plan['tokenizer_sha256']
    tokenizer = read(tokenizer_path)
    pieces = token_bytes(tokenizer)
    decode, _ = decoder(tokenizer)
    frozen = read(ROOT/'integration/rebalance_easysteer/configs/final_results_20260909.json')
    think_end = frozen['benchmarks'][0]['protocol']['dynamic_params']['think_end_token_id']
    audit30 = read(ROOT/'integration/rebalance_easysteer/configs/calibration_label_evidence30_20260910.json')
    boundary20 = read(ROOT/'integration/rebalance_easysteer/configs/boundary_ablation20_plan_20260910.json')
    reviewed30 = {q['calibration_index'] for q in audit30['questions']}
    boundary_indices = {q['calibration_index'] for q in boundary20['cases']}
    assert len(reviewed30)==30 and len(boundary_indices)==20 and not reviewed30 & boundary_indices
    results, details, tail_holds = [], [], []
    archive_path = ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        assert manifest['temperature']==0 and manifest['count']==500 and manifest['max_tokens']==16000
        with archive.extractfile('generations.jsonl') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == plan['source_sha256']
        for index, line in enumerate(archive.extractfile('generations.jsonl')):
            row = json.loads(line)
            assert row['train_index']==manifest['train_indices'][index]
            ids = row['token_ids']
            assert len(ids)<=16000
            decoded = decode(ids, skip_special=True)
            if decoded != row['text']:
                suffix = decoded[len(row['text']):]
                assert row['finish_reason']=='length' and decoded.startswith(row['text'])
                assert '\ufffd' in suffix and all(c.isspace() or c=='\ufffd' for c in suffix)
                tail_holds.append(index)
            monitor = replay(ids, pieces, think_end)
            # Independent byte-stream coverage of every completed paragraph.
            end = ids.index(think_end) if think_end in ids else len(ids)
            stream = b''.join(pieces[i] for i in ids[:end])
            complete = [x.strip() for x in stream.split(b'\n\n')[:-1] if x.strip()]
            assert [hashlib.sha256(x).hexdigest() for x in complete] == [u['normalized_bytes_sha256'] for u in monitor.units]
            byte_ends=[]
            total=0
            for token in ids[:end]:
                total+=len(pieces[token]);byte_ends.append(total)
            episodes=[episode_record(e,monitor,byte_ends) for e in monitor.episodes]
            first=episodes[0] if episodes else None
            group='reviewed30' if index in reviewed30 else 'boundary20' if index in boundary_indices else 'remaining450'
            record=dict(calibration_index=index,train_index=row['train_index'],exposure_group=group,
                finish_reason=row['finish_reason'],total_tokens=len(ids),thinking_tokens=end,
                thinking_closed=think_end in ids,completed_paragraphs=len(monitor.units),
                alert=bool(first),first_alert=first,episode_count=len(episodes),
                max_completed_copies=max((e['completed_copies'] for e in episodes),default=0),
                observed_tokens_after_first_alert=len(ids)-first['detected_token'] if first else None,
                observed_thinking_after_first_alert=end-first['detected_token'] if first else None)
            results.append(record)
            details.append(dict(**record,problem=row['problem'],text=row['text'],units=monitor.units,
                                unfinished_tail=bytes(monitor.buffer).decode('utf-8',errors='replace'),episodes=episodes))
    assert len(results)==500 and len({r['train_index'] for r in results})==500
    counts={}
    for group in ('all500','reviewed30','boundary20','remaining450'):
        subset=[r for r in results if group=='all500' or r['exposure_group']==group]
        counts[group]={reason:dict(count=sum(r['finish_reason']==reason for r in subset),
            alerted=sum(r['finish_reason']==reason and r['alert'] for r in subset)) for reason in ('stop','length')}
    stopped_alerts=sorted(r['calibration_index'] for r in results if r['finish_reason']=='stop' and r['alert'])
    review_indices=sorted(random.Random(20260911).sample(stopped_alerts,min(12,len(stopped_alerts))))
    missed_caps=[r['calibration_index'] for r in results if r['finish_reason']=='length' and not r['alert']]
    report=dict(status='completed_cpu_shadow_replay',date='2026-09-11',question_count=500,
        new_generations=0,model_forwards=0,gpu_tasks=0,plan=plan,
        provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256=plan['source_sha256'],tokenizer_sha256=sha(tokenizer_path),
            script_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),
            manifest=manifest,decode_tail_holds=tail_holds),
        outcome_cross_tab=counts,review_sample=dict(seed=20260911,alerted_stopped=review_indices,all_unalerted_capped=missed_caps),
        records=results,cpu_seconds=time.perf_counter()-started,
        limitations=['Capped/not capped is not semantic loop truth. No precision or recall is inferred from this table.',
          'All 500 trained the old calibration artifacts; remaining450 is unreviewed exposure accounting, not an independent validation split.',
          'Remaining observed tokens are hypothetical opportunity only, not saved tokens, retained accuracy or measured speedup.',
          'Only exact adjacent completed-block repeats are detected; rephrased or interrupted loops can be missed.',
          'Rule and review sampling were fixed before replay. Alerts do not trigger stopping, select an answer or change steering.'])
    save(output/'summary.json',report)
    save(output/'details.json',details)
    print(json.dumps(dict(counts=counts,review_sample=report['review_sample'],cpu_seconds=report['cpu_seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'integration/rebalance_easysteer/configs/cycle_monitor_plan_20260911.json')
    args=parser.parse_args()
    run(args.output,args.plan)
