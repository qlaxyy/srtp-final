"""Inspect long-suffix eligibility on saved paths, without generating answers."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from replay_cycle_monitor import token_bytes
from suffix_penalty import SuffixPenaltyConfig, SuffixMonitor, checks, brute_force


def audit(ids, breakers, config, end_id):
    monitor = SuffixMonitor(breakers, config); events = []; thinking = 0
    for position, token in enumerate(ids):
        if token == end_id: break
        witness = monitor.saved_next_token_witness(token)
        if witness is not None:
            length, previous = witness
            events.append((position, length, previous))
        monitor.feed(token); thinking += 1
    return dict(thinking_tokens=thinking, eligible_saved_next_tokens=len(events), events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); out = a.output.resolve(); require(not out.exists(), 'Immutable output exists')
    started = time.perf_counter(); unit = checks(); out.mkdir(parents=True)
    plan_path = ROOT/BASE/'configs/suffix_penalty_20260912.json'; plan = read(plan_path); fixed = plan['fixed_adaptation']
    config = SuffixPenaltyConfig(fixed['allowed_length'], fixed['history_tokens'], fixed['multiplier'], fixed['base'], fixed['maximum_logit_penalty'])
    old_plan_path = ROOT/BASE/'configs/seal_comparison130_20260912/plan.json'; old_plan = read(old_plan_path)
    tokenizer_path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    require(sha(tokenizer_path) == old_plan['model_files_sha256']['tokenizer.json'], 'Tokenizer changed')
    tokenizer = read(tokenizer_path); pieces = token_bytes(tokenizer)
    breaker_bytes = [s.encode('utf-8') for s in fixed['breaker_bytes']]
    breakers = {i for i, piece in pieces.items() if any(marker in piece for marker in breaker_bytes)}
    breakers |= {t['id'] for t in tokenizer['added_tokens'] if t['special']}
    dynamic_path = ROOT/'.codex_work/overnight_research_20260912/seal_comparison_all_20260912/seal_comparison130_20260912/math_train_original_dynamic.json'
    dynamic = read(dynamic_path); ledger = read(dynamic_path.parent/'ledger.json')
    require(sha(dynamic_path) == ledger['files_sha256']['math_train_original_dynamic'], 'Dynamic source changed')
    require(dynamic['status'] == 'completed' and dynamic['arm'] == 'original_dynamic' and len(dynamic['records']) == 100, 'Wrong dynamic group')
    require(dynamic['plan_sha256'] == sha(old_plan_path, source=True), 'Dynamic plan changed')
    population = {}; sparse = []; source_hashes = dict(dynamic=sha(dynamic_path)); raw_dynamic = dynamic['records']
    for name in ['dynamic', 'calibration']:
        summaries = []; digest = hashlib.sha256(); cap_count = 0
        if name == 'calibration':
            archive = tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz')
            stream = archive.extractfile('generations.jsonl')
        else: stream = raw_dynamic
        with (out/(name+'_progress.jsonl')).open('x', encoding='utf-8', newline='\n') as progress:
            for q, entry in enumerate(stream):
                if name == 'calibration': digest.update(entry); row = json.loads(entry)
                else: row = entry
                require(len(row['token_ids']) <= 16000 and ((row['finish_reason'] == 'length') == (len(row['token_ids']) == 16000)), 'Source cap/count mismatch')
                observed = audit(row['token_ids'], breakers, config, 151649)
                if name == 'dynamic': require(observed['thinking_tokens'] == row['thinking_tokens'], 'Thinking count changed')
                cap_count += row['finish_reason'] == 'length'
                events = observed.pop('events')
                sparse.extend((0 if name == 'dynamic' else 1, q, *event) for event in events)
                result = dict(question=q, capped=row['finish_reason'] == 'length', **observed,
                    first_event=list(events[0]) if events else None)
                summaries.append(result); progress.write(json.dumps(result)+'\n'); progress.flush()
                if (q+1) % 100 == 0: print(dict(population=name, completed=q+1), flush=True)
        if name == 'calibration':
            archive.close(); require(q == 499 and digest.hexdigest() == '4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3', 'Original answers changed')
            source_hashes[name] = digest.hexdigest()
        total = sum(r['thinking_tokens'] for r in summaries); eligible = sum(r['eligible_saved_next_tokens'] for r in summaries)
        population[name] = dict(questions=len(summaries), caps=cap_count, thinking_tokens=total,
            eligible_saved_next_tokens=eligible, eligible_fraction=eligible/total,
            affected_questions=sum(r['eligible_saved_next_tokens'] > 0 for r in summaries), per_question=summaries)
    candidates = [r for r in population['dynamic']['per_question'] if r['first_event'] is not None]
    gate = plan['CPU_gate']; selected = random.Random(gate['manual_audit_seed']).sample(candidates, min(len(candidates), gate['manual_audit_max_questions']))
    packets = []
    for number, summary in enumerate(selected):
        row = raw_dynamic[summary['question']]; pos, length, previous = summary['first_event']; ids = row['token_ids']
        reference = brute_force(ids[:pos], breakers, config)
        require(reference[ids[pos]] == (length, previous), 'Manual witness differs from independent backward search')
        def decode(tokens): return b''.join(pieces[t] for t in tokens).decode('utf-8', errors='replace')
        packets.append(dict(case_id=f'SUFFIX{number+1:02d}', problem=row['problem'],
            arrived_prefix_tokens=pos, suffix_length=length, proposed_next_token=decode([ids[pos]]),
            proposed_penalty=config.penalty(length), current_suffix=decode(ids[pos-length:pos]),
            earlier_witness_span=decode(ids[previous-length+1:previous+2]),
            earlier_witness_token_interval=[previous-length+1, previous+2],
            causal_prefix=decode(ids[:pos]),
            required_judgment='redundant_continuation / useful_reuse / uncertain; cite prefix evidence; do not view final answer or grade'))
    save(out/'manual_packets.json', packets)
    save(out/'manual_key.json', [dict(case_id=p['case_id'], question=s['question'], train_index=old_plan['math_train_indices'][s['question']]) for p, s in zip(packets, selected, strict=True)])
    np.savez_compressed(out/'eligible_events.npz', events=np.asarray(sparse, dtype=np.int32).reshape(-1, 5))
    objective_pass = (population['dynamic']['affected_questions'] >= gate['minimum_dynamic_questions_with_actual_next_token_penalized'] and
        population['dynamic']['eligible_fraction'] >= gate['minimum_dynamic_thinking_token_fraction_of_actual_next_tokens_penalized'])
    result = dict(status='completed_CPU_saved_prefix_eligibility_pending_content_review',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True),
        monitor_sha256=sha(Path(__file__).with_name('suffix_penalty.py'), source=True), source_answer_sha256=source_hashes,
        tokenizer_sha256=sha(tokenizer_path), breaker_count=len(breakers), breaker_ids=sorted(breakers), CPU_checks=unit,
        populations=population, objective_gate_passed=bool(objective_pass), manual_questions=len(packets),
        passes_fixed_gate=None, decision='finish_fixed_manual_screen' if objective_pass else 'stop_objective_gate_failed_no_GPU',
        files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}, cpu_seconds=time.perf_counter()-started,
        model_loads=0, GPU_calls=0, new_answers=0,
        interpretation='All counts are eligibility on saved original prefixes. No penalties were used to generate them; counts and future remaining tokens are not compression or online effect.')
    save(out/'audit.json', result)
    print(dict(status=result['status'], objective_gate_passed=objective_pass,
        populations={k:{key:v[key] for key in ['questions','caps','thinking_tokens','eligible_saved_next_tokens','eligible_fraction','affected_questions']} for k,v in population.items()}, cpu_seconds=result['cpu_seconds']))


if __name__ == '__main__': main()
