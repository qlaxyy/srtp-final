"""CPU-only causal atom-cycle witnesses; no generation, stopping or steering."""
import argparse
from bisect import bisect_left
import codecs
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tarfile
import time
from types import SimpleNamespace

from audit_calibration_labels import ROOT, read, save, sha, decoder
from audit_control_alignment import load_controller, scalar_reference
from replay_cycle_monitor import token_bytes


class AtomCycleMonitor:
    """Consume arrived bytes; token positions count generated tokens from one."""
    HASH_BASE = 1000003
    HASH_MASK = (1 << 64) - 1

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.byte_ends = []
        self.byte_cursor = 0
        self.atoms = []
        self.symbols = []
        self.intern = {}
        self.occurrences = {}
        self.hashes = [0]
        self.powers = [1]
        self.episodes = []
        self.active = None
        self.pending = []
        self.pending_kind = None
        self.pending_start = self.pending_end = 0
        self.number_separator = None

    def _substring_hash(self, start, end):
        return (self.hashes[end] - self.hashes[start]*self.powers[end-start]) & self.HASH_MASK

    def _emit(self, text, kind, start, end, token, events):
        self.atoms.append(dict(text=text, kind=kind, start_byte=start, end_byte=end, closed_token=token))
        symbol = self.intern.setdefault((kind, text), len(self.intern)+1)
        self.symbols.append(symbol)
        self.hashes.append((self.hashes[-1]*self.HASH_BASE + symbol) & self.HASH_MASK)
        self.powers.append(self.powers[-1]*self.HASH_BASE & self.HASH_MASK)
        n = len(self.symbols)
        previous = self.occurrences.setdefault(symbol, [])
        if self.active is not None:
            period = self.active['period_atoms']
            if symbol == self.symbols[-1-period]:
                self.active['observed_end_atom'] = n
                self.active['completed_copies'] = (n-self.active['start_atom'])//period
                previous.append(n-1)
                return
            self.active['closed_on_mismatch_token'] = token
            self.active = None
        for index in reversed(previous):
            period = n-1-index
            if 2*period > n:
                break
            if period == 1:
                continue
            start_atom = n-2*period
            if self._substring_hash(start_atom, n-period) != self._substring_hash(n-period, n):
                continue
            block = self.symbols[n-period:n]
            if len(set(block)) < 2 or self.symbols[start_atom:n-period] != block:
                continue
            event = dict(start_atom=start_atom, period_atoms=period, detected_end_atom=n,
                         detected_token=token, detected_last_atom_end_byte=end)
            self.active = dict(**event, observed_end_atom=n, completed_copies=2,
                               closed_on_mismatch_token=None)
            self.episodes.append(self.active)
            events.append(dict(event))
            break
        previous.append(n-1)

    def _flush_pending(self, token, events):
        self._emit(''.join(self.pending), self.pending_kind, self.pending_start, self.pending_end, token, events)
        self.pending = []
        self.pending_kind = None

    def _character(self, char, start, end, token, events):
        if self.pending_kind == 'word':
            if char.isalnum() or char == '_':
                self.pending.append(char)
                self.pending_end = end
                return
            self._flush_pending(token, events)
        elif self.pending_kind == 'number':
            if self.number_separator is not None:
                separator, sep_start, sep_end = self.number_separator
                self.number_separator = None
                if char.isdigit():
                    self.pending.extend([separator, char])
                    self.pending_end = end
                    return
                self._flush_pending(token, events)
                self._emit(separator, 'symbol', sep_start, sep_end, token, events)
            elif char.isdigit():
                self.pending.append(char)
                self.pending_end = end
                return
            elif char in '.,':
                self.number_separator = (char, start, end)
                return
            else:
                self._flush_pending(token, events)
        if char.isspace():
            return
        if char.isalpha() or char == '_' or char.isdigit():
            self.pending_kind = 'number' if char.isdigit() else 'word'
            self.pending = [char]
            self.pending_start, self.pending_end = start, end
        else:
            self._emit(char, 'symbol', start, end, token, events)

    def feed(self, piece, token_end):
        self.byte_ends.append((self.byte_ends[-1] if self.byte_ends else 0)+len(piece))
        text = self.decoder.decode(piece, final=False)
        events = []
        for char in text:
            start = self.byte_cursor
            self.byte_cursor += len(char.encode('utf-8'))
            self._character(char, start, self.byte_cursor, token_end, events)
        return events


def replay_atoms(ids, pieces, think_end):
    monitor = AtomCycleMonitor()
    for n, token in enumerate(ids, 1):
        if token == think_end:
            break
        monitor.feed(pieces[token], n)
    return monitor


def episode_record(event, monitor, stream):
    start, period = event['start_atom'], event['period_atoms']
    atoms = monitor.atoms
    first = atoms[start]
    block_end = atoms[start+period-1]['end_byte']
    second_start = atoms[start+period]['start_byte']
    detected_end = atoms[start+2*period-1]['end_byte']
    complete_end = start+event['completed_copies']*period
    end_byte = atoms[complete_end-1]['end_byte']
    token_start = bisect_left(monitor.byte_ends, first['start_byte']+1)
    token_stop = bisect_left(monitor.byte_ends, end_byte)+1
    normalized = [(a['kind'],a['text']) for a in atoms[start:start+period]]
    assert normalized == [(a['kind'],a['text']) for a in atoms[start+period:start+2*period]]
    return dict(**event, block_byte_span=[first['start_byte'],block_end],
        second_block_byte_span=[second_start,detected_end],
        completed_repeat_byte_span=[first['start_byte'],end_byte],
        completed_repeat_token_span=[token_start,token_stop],
        completed_repeat_span_tokens=token_stop-token_start,
        normalized_block_sha256=hashlib.sha256(json.dumps(normalized,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        block_text=stream[first['start_byte']:block_end].decode('utf-8'),
        second_block_text=stream[second_start:detected_end].decode('utf-8'),
        block_contains_number=any(a['kind']=='number' for a in atoms[start:start+period]),
        block_contains_word=any(a['kind']=='word' for a in atoms[start:start+period]))


def run(output, plan_path):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    plan = read(plan_path)
    save(output/'plan.json', plan)
    configs = ROOT/'integration/rebalance_easysteer/configs'
    old_path = configs/'cycle_monitor500_20260911.json'
    old = read(old_path)
    frozen = read(configs/'final_results_20260909.json')
    fit = read(configs/'auto_code_v2_1p5b_20260908.json')
    p = SimpleNamespace(**fit['parameters'])
    assert all(frozen['benchmarks'][0]['protocol']['dynamic_params'][k]==v for k,v in fit['parameters'].items())
    runtime_path = 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
    blob = subprocess.check_output(['git','show','HEAD:'+runtime_path],cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest()==frozen['source_sha256'][runtime_path]
    assert blob.decode().replace('\r\n','\n')==(ROOT/runtime_path).read_text(encoding='utf-8')
    functions = load_controller()
    constants = functions['_curve_constants'](p.q25c,p.q75c,p.low_val_1,'cpu','float64',p.curve_tau)
    params = frozen['benchmarks'][0]['protocol']['dynamic_params']
    boundaries, think_end = set(params['boundary_token_ids']), params['think_end_token_id']
    tokenizer_path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    assert sha(tokenizer_path)==plan['tokenizer_sha256']
    tokenizer = read(tokenizer_path)
    pieces = token_bytes(tokenizer)
    decode, _ = decoder(tokenizer)
    assert {i for t,i in tokenizer['model']['vocab'].items() if 'ĊĊ' in t}==boundaries
    records, details, decode_holds = [], [], []
    archive_path = ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        with archive.extractfile('generations.jsonl') as source:
            assert hashlib.file_digest(source,'sha256').hexdigest()==plan['source_sha256']==old['provenance']['source_sha256']
        for index, line in enumerate(archive.extractfile('generations.jsonl')):
            row = json.loads(line)
            ids = row['token_ids']
            prior = old['records'][index]
            assert row['train_index']==manifest['train_indices'][index]==prior['train_index']
            assert row['finish_reason']==prior['finish_reason'] and len(ids)==prior['total_tokens']
            decoded = decode(ids,skip_special=True)
            if decoded!=row['text']:
                suffix=decoded[len(row['text']):]
                assert row['finish_reason']=='length' and decoded.startswith(row['text'])
                assert '\ufffd' in suffix and all(c.isspace() or c=='\ufffd' for c in suffix)
                decode_holds.append(index)
            monitor = AtomCycleMonitor()
            step_sum = 0.0
            step_count = 0
            previous_mean = None
            boundary_count = positive_count = active_count = 0
            opportunities = []
            for n, token in enumerate(ids,1):
                if token==think_end:
                    break
                monitor.feed(pieces[token],n)
                if token not in boundaries:
                    step_sum += math.exp(row['logprobs'][n-1])
                    step_count += 1
                elif step_count:
                    confidence = step_sum/step_count
                    variance = 0.0 if previous_mean is None else (confidence-previous_mean)**2/4
                    coef = scalar_reference(confidence,variance,p,constants)['coefficient']
                    boundary_count += 1
                    positive_count += coef>1e-6
                    active_count += monitor.active is not None
                    if monitor.active is not None and coef>1e-6:
                        opportunities.append(dict(boundary_token_1based=n,coefficient=coef,confidence=confidence,variance=variance,
                            episode_start_atom=monitor.active['start_atom'],episode_detected_token=monitor.active['detected_token']))
                    previous_mean = confidence
                    step_sum, step_count = 0.0, 0
            thinking_tokens = ids.index(think_end) if think_end in ids else len(ids)
            stream = b''.join(pieces[t] for t in ids[:thinking_tokens])
            assert monitor.byte_cursor+len(monitor.decoder.getstate()[0])==len(stream)
            for atom in monitor.atoms:
                assert stream[atom['start_byte']:atom['end_byte']].decode('utf-8')==atom['text']
            episodes = [episode_record(e,monitor,stream) for e in monitor.episodes]
            first = episodes[0] if episodes else None
            longest = max(episodes,key=lambda e:e['completed_repeat_span_tokens'],default=None)
            record = dict(calibration_index=index,train_index=row['train_index'],exposure_group=prior['exposure_group'],
                finish_reason=row['finish_reason'],total_tokens=len(ids),thinking_tokens=thinking_tokens,
                old_paragraph_alert=prior['alert'],atom_alert=bool(first),union_alert=prior['alert'] or bool(first),
                atom_count=len(monitor.atoms),episode_count=len(episodes),first_alert=first,longest_episode=longest,
                observed_tokens_after_first_alert=len(ids)-first['detected_token'] if first else None,
                runtime_boundaries=boundary_count,original_positive_boundaries=positive_count,
                atom_active_boundaries=active_count,positive_gate_opportunities=len(opportunities))
            records.append(record)
            details.append(dict(**record,problem=row['problem'],thinking_text=stream.decode('utf-8',errors='replace'),
                text=row['text'],episodes=episodes,positive_opportunities=opportunities,
                incomplete_atom=''.join(monitor.pending),pending_numeric_separator=monitor.number_separator,
                incomplete_utf8_hex=monitor.decoder.getstate()[0].hex()))
            if (index+1)%100==0:
                print(f'Replayed {index+1}/500 saved answers',flush=True)
    assert len(records)==500
    counts={}
    for group in ('all500','reviewed30','boundary20','remaining450'):
        subset=[q for q in records if group=='all500' or q['exposure_group']==group]
        counts[group]={}
        for reason in ('stop','length'):
            rows=[q for q in subset if q['finish_reason']==reason]
            counts[group][reason]=dict(count=len(rows),paragraph_alerted=sum(q['old_paragraph_alert'] for q in rows),
                atom_alerted=sum(q['atom_alert'] for q in rows),union_alerted=sum(q['union_alert'] for q in rows),
                questions_with_positive_opportunity=sum(q['positive_gate_opportunities']>0 for q in rows),
                positive_gate_opportunities=sum(q['positive_gate_opportunities'] for q in rows))
    new_stopped=sorted(q['calibration_index'] for q in records if q['finish_reason']=='stop' and q['atom_alert'] and not q['old_paragraph_alert'])
    selected=sorted(random.Random(plan['review']['seed']).sample(new_stopped,min(12,len(new_stopped))))
    report=dict(status='completed_cpu_atom_replay_pending_scoped_review',date=plan['date'],plan=plan,
        question_count=500,new_generations=0,model_forwards=0,gpu_tasks=0,
        provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            script_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),source_sha256=plan['source_sha256'],
            tokenizer_sha256=sha(tokenizer_path),earlier_report_sha256=sha(old_path),frozen_runtime_blob_sha256=hashlib.sha256(blob).hexdigest(),
            manifest=manifest,decode_tail_holds=decode_holds),
        outcome_cross_tab=counts,review_sample=dict(new_stopped=selected,known_stopped=[274,348,469],
            earlier_unalerted_caps=old['review_sample']['all_unalerted_capped']),
        records=records,elapsed_seconds=time.perf_counter()-started,
        limitations=['Outcome cross-tabs are not semantic precision/recall.',
            'The rule uses prior failure review; all500 are development data, not independent validation.',
            'Ignoring whitespace and comparing complete atoms admits legitimate short restatements and calculations.',
            'No persistence threshold or action trigger has been validated. Single-atom repetition is intentionally not captured.',
            'Later episode growth and remaining observed tokens are not evidence available at the first alert, nor actual saved tokens.',
            'Positive opportunities are float64 projections on unsteered prefixes; runtime overhead and causal gate benefit remain unmeasured.'])
    save(output/'summary.json',report)
    save(output/'details.json',details)
    print(json.dumps(dict(counts=counts,review_sample=report['review_sample'],elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'integration/rebalance_easysteer/configs/cycle_atoms_plan_20260911.json')
    args=parser.parse_args()
    run(args.output,args.plan)
