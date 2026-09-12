"""SEAL lexical labels on the original ReBalance support, without model loads."""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import ROOT, BASE, read, save, sha, require, moments, read_vector, write_vector, cosine
from replay_cycle_monitor import token_bytes


def author_lexical_rule(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_index')
    names = {'check_words', 'check_prefix', 'swicth_words', 'switch_prefix'}
    lists = {}
    for child in node.body:
        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and child.targets[0].id in names:
            lists[child.targets[0].id] = ast.literal_eval(child.value)
    require(set(lists) == names, 'Official lexical constants incomplete')
    def label(text):
        text = text.strip(' ').strip('\n').lower()
        if any(text.startswith(p.lower()) for p in lists['check_prefix']) or any(w.lower() in text for w in lists['check_words']): return 1
        if any(text.startswith(p.lower()) for p in lists['switch_prefix']) or any(w.lower() in text for w in lists['swicth_words']): return 2
        return 0
    return label, lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); out = a.output.resolve(); require(not out.exists(), 'Immutable output exists')
    started = time.perf_counter()
    plan_path = ROOT/BASE/'configs/lexical_direction_20260912.json'; plan = read(plan_path)
    work = ROOT/'.codex_work/overnight_research_20260912'; backup = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    official_plan = read(work/'seal_prepared/replay_plan.json')
    source = work/'upstream/SEAL/hidden_analysis.py'
    require(sha(source, source=True) == official_plan['author_label_source_sha256'], 'Official source changed')
    label, lists = author_lexical_rule(source)
    cases = {'Wait, another way':1, 'Alternatively, verify this':1, 'another approach':2,
             'think differently':0, 'think differenly':2, 'Waiter':1, 'x = 2':0}
    require({s: label(s) for s in cases} == cases, 'Official substring/precedence rules changed')
    tokenizer_path = ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'
    require(sha(tokenizer_path) == official_plan['model_files_sha256']['tokenizer.json'], 'Tokenizer changed')
    pieces = token_bytes(read(tokenizer_path)); steps = read(backup/'steps.json'); original_fit = read(backup/'fit.json')
    selected = read(backup/'selected_layer.json')
    source_hashes = {name: sha(backup/name) for name in ['steps.json', 'fit.json', 'auto_vector.pt', 'layer_21.npy']}
    require(all(digest == official_plan['original_files_sha256'][name] for name, digest in source_hashes.items()), 'Original calibration changed')
    positions_path = work/'seal_prepared/positions.json'; require(sha(positions_path) == official_plan['positions_sha256'], 'Official position labels changed')
    official_maps = read(positions_path)
    grouped = [[] for _ in range(500)]
    for i, step in enumerate(steps): grouped[step['question']].append((i, step))
    labels = np.empty(len(steps), dtype=np.int8); digest = hashlib.sha256(); matched = 0; unmatched = []
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as tar:
        for q, line in enumerate(tar.extractfile('generations.jsonl')):
            digest.update(line); row = json.loads(line); ids = row['token_ids']; prompt = len(row['prompt_token_ids'])
            official = dict(zip(official_maps[q]['positions'], official_maps[q]['classes'], strict=True))
            for i, step in grouped[q]:
                text = b''.join(pieces[token] for token in ids[step['start']:step['stop']]).decode('utf-8', errors='replace')
                labels[i] = label(text); predecessor = prompt+step['start']-1
                if predecessor in official:
                    require(labels[i] == official[predecessor], f'Original/official step label differs at{q}/{i}')
                    matched += 1
                else:
                    require(step['start'] == 0, 'Non-first original step lacks official counterpart')
                    unmatched.append(i)
    require(q == 499 and digest.hexdigest() == official_plan['generations_sha256'], 'Saved answers changed')
    require(matched == 83508 and len(unmatched) == 500, 'Unexpected step support')
    question = np.array([s['question'] for s in steps]); c = np.array([s['confidence'] for s in steps]); lexical = np.array([s['lexical_hit'] for s in steps])
    old_over = lexical | (c < original_fit['parameters']['q25c']); old_under = ~lexical & (c > original_fit['parameters']['q75c'])
    reflect_transition = labels != 0; execution = labels == 0
    train = np.isin(question, selected['training_questions']); development = np.isin(question, selected['validation_questions'])
    masks = dict(RT=reflect_transition, E=execution, train_RT=reflect_transition & train, train_E=execution & train,
                 dev_RT=reflect_transition & development, dev_E=execution & development)
    x = np.load(backup/'layer_21.npy', mmap_mode='r'); require(x.shape == (84008, 1536), 'Wrong original states')
    stats = moments(x, masks)
    direction = stats['RT']['mean']-stats['E']['mean']
    original = read_vector(backup/'auto_vector.pt', backup/'auto_vector.pt').astype(np.float64)
    candidate = direction * (np.linalg.norm(original)/np.linalg.norm(direction))
    train_d = stats['train_RT']['mean']-stats['train_E']['mean']; dev_d = stats['dev_RT']['mean']-stats['dev_E']['mean']
    agreement = cosine(train_d, dev_d); change = np.linalg.norm(candidate-original)/np.linalg.norm(original)
    questions_per_class = {name: len(np.unique(question[mask])) for name, mask in [('RT', reflect_transition), ('E', execution)]}
    gate = plan['CPU_gate']; passed = (min(questions_per_class.values()) >= gate['minimum_questions_per_binary_class'] and
        agreement >= gate['minimum_original400_vs100_direction_cosine'] and
        gate['minimum_relative_vector_change_at_matched_norm'] <= change <= gate['maximum_relative_vector_change_at_matched_norm'])
    content_path = ROOT/'.codex_work/label_audit_30_20260910/unmasked_audit.json'
    content = read(content_path); joined = []
    # Resolve by exact saved question/token interval, not by order in review packets.
    lookup = {(s['question'], s['start'], s['stop']): i for i, s in enumerate(steps)}
    content_rows = next(v for v in content.values() if isinstance(v, list) and len(v) == 76 and isinstance(v[0], dict) and 'review' in v[0])
    for row in content_rows:
        index = lookup[(row['calibration_index'], row['start'], row['stop'])]
        joined.append(dict(case_id=row['case_id'], question=row['calibration_index'], original_label=row['original_label'],
            new_class=['execution', 'reflection', 'transition'][labels[index]], content_category=row['review']['category'], math_validity=row['review']['math_validity']))
    out.mkdir(parents=True); np.save(out/'labels.npy', labels)
    summary = dict(status='completed_CPU_lexical_direction_fit_not_generation',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        plan_sha256=sha(plan_path, source=True), script_sha256=sha(Path(__file__), source=True), source_sha256=source_hashes,
        author_source_sha256=sha(source, source=True), official_source=official_plan['original_source'], official_label_constants=lists,
        exact_step_support=len(steps), exact_official_shared_labels_checked=matched, first_steps_labeled_with_same_rule=len(unmatched),
        class_counts=dict(Counter(['execution', 'reflection', 'transition'][i] for i in labels)), class_questions=questions_per_class,
        original_binary_label_overlaps={name: dict(RT=int((mask & reflect_transition).sum()), E=int((mask & execution).sum()))
            for name, mask in [('over', old_over), ('under', old_under), ('unselected', ~old_over & ~old_under)]},
        raw_contrast_norm=float(np.linalg.norm(direction)), matched_vector_norm=float(np.linalg.norm(candidate)),
        original_vector_cosine=cosine(direction, original), relative_vector_change=float(change), group_direction_cosine=agreement,
        content_judgment_source_sha256=sha(content_path), joined_content_diagnostics=joined,
        passes_fixed_gate=bool(passed), decision='prepare_fresh_GPU_screen' if passed else 'stop_no_GPU_no_label_search',
        label_array_sha256=sha(out/'labels.npy'), CPU_rule_checks=len(cases), cpu_seconds=time.perf_counter()-started,
        model_loads=0, new_answers=0, GPU_calls=0, limitations=plan['risks'])
    if passed:
        write_vector(out/'auto_vector.pt', candidate.astype(np.float32), backup/'auto_vector.pt')
        fit = copy.deepcopy(original_fit); fit['version'] = 'auto-code-v2-SEAL-lexical-same-support'
        fit['vector_sha256'] = sha(out/'auto_vector.pt'); fit['vector_norm'] = float(np.linalg.norm(candidate.astype(np.float32)))
        fit['positives'] = int(reflect_transition.sum()); fit['negatives'] = int(execution.sum())
        fit['method'] = 'SEAL lexical RT minusE on original84008 first-content states, matched originalnorm; all original dynamic parameters retained. Hybrid, not staticSEAL.'
        fit['candidate_provenance'] = dict(hypothesis_sha256=summary['plan_sha256'], original_fit_sha256=source_hashes['fit.json'],
            labels_sha256=summary['label_array_sha256'], selected_layer_unchanged=True, support_unchanged=True)
        save(out/'fit.json', fit); summary['candidate_files_sha256'] = {name: sha(out/name) for name in ['auto_vector.pt', 'fit.json']}
    save(out/'audit.json', summary)
    print({k: summary[k] for k in ['status', 'class_counts', 'exact_official_shared_labels_checked', 'original_vector_cosine', 'relative_vector_change', 'group_direction_cosine', 'passes_fixed_gate', 'decision', 'cpu_seconds']})


if __name__ == '__main__': main()
