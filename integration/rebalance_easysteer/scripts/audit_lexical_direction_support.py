"""Describe the fixed lexical vector's token/question dependence; never refit it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from mechanism_candidates import ROOT, read, save, sha, require, cosine, read_vector
from replay_cycle_monitor import token_bytes


def decompose(x, labels, groups, selected_rows=None, reference=None):
    """D = prefix-composition component + within-prefix residual component.

    Means use original step weights. A projection fraction is not explained
    variance or evidence of semantic causation, and can lie outside [0, 1].
    """
    if selected_rows is not None:
        labels = labels[selected_rows]; groups = groups[selected_rows]
    names, inverse = np.unique(groups, return_inverse=True); size = len(names)
    counts = np.zeros((2, size), dtype=np.int64)
    sums = np.zeros((2, size, x.shape[1]), dtype=np.float64)
    for start in range(0, len(labels), 2048):
        stop = min(start+2048, len(labels)); cls = labels[start:stop].astype(np.int64)
        index = slice(start, stop) if selected_rows is None else selected_rows[start:stop]
        idx = inverse[start:stop]; block = np.asarray(x[index], dtype=np.float64)
        np.add.at(counts, (cls, idx), 1)
        np.add.at(sums, (cls, idx), block)
    total_count = counts.sum(axis=1); total_sum = sums.sum(axis=1)
    direction = total_sum[1]/total_count[1]-total_sum[0]/total_count[0]
    means = sums.sum(axis=0)/counts.sum(axis=0)[:, None]
    proportions = counts/total_count[:, None]
    composition = (proportions[1]-proportions[0]) @ means
    # Independent direct within-group centering, not D minus composition.
    centered = sums-counts[:, :, None]*means
    residual = centered[1].sum(axis=0)/total_count[1]-centered[0].sum(axis=0)/total_count[0]
    require(np.allclose(direction, composition+residual, rtol=1e-11, atol=1e-11), 'Decomposition identity failed')
    norm = np.linalg.norm(direction)
    leave_one_out = []
    for i, name in enumerate(names):
        remaining = total_count-counts[:, i]
        if np.all(remaining > 0):
            sums_remaining = total_sum-sums[:, i]
            other = sums_remaining[1]/remaining[1]-sums_remaining[0]/remaining[0]
            leave_one_out.append(dict(group=str(name), steps=int(counts[:, i].sum()),
                RT_steps=int(counts[1, i]), E_steps=int(counts[0, i]),
                direction_cosine=cosine(direction, other), relative_norm=float(np.linalg.norm(other)/norm)))
    return dict(groups=len(names), raw_contrast_norm=float(norm),
        supplied_reference_cosine=cosine(direction, reference) if reference is not None else None,
        composition_projection_fraction=float(composition@direction/(norm*norm)),
        composition_norm_over_full=float(np.linalg.norm(composition)/norm),
        residual_norm_over_full=float(np.linalg.norm(residual)/norm),
        residual_full_cosine=cosine(residual, direction),
        within_group_mixed_class_steps=int(counts[:, np.all(counts>0, axis=0)].sum()),
        exact_additive_reconstruction_max_error=float(abs(direction-composition-residual).max()),
        leave_one_group_out=leave_one_out)


def cross_group_decomposition(x, labels, groups, question, training, development, selected_rows=None):
    indices = np.arange(len(x)) if selected_rows is None else selected_rows
    labels = labels[indices]; groups = groups[indices]; question = question[indices]
    names, inverse = np.unique(groups, return_inverse=True); size = len(names)
    train = np.isin(question, training); dev = np.isin(question, development)
    require(np.all(train ^ dev), 'Question groups overlap')
    pooled_count = np.zeros(size, dtype=np.int64); pooled_sum = np.zeros((size, x.shape[1]))
    dev_count = np.zeros((2, size), dtype=np.int64); dev_sum = np.zeros((2, x.shape[1]))
    for start in range(0, len(indices), 2048):
        end = min(start+2048, len(indices)); block = np.asarray(x[indices[start:end]], dtype=np.float64)
        t = train[start:end]; d = dev[start:end]; group = inverse[start:end]; cls = labels[start:end].astype(int)
        np.add.at(pooled_count, group[t], 1); np.add.at(pooled_sum, group[t], block[t])
        np.add.at(dev_count, (cls[d], group[d]), 1); np.add.at(dev_sum, cls[d], block[d])
    present = pooled_count > 0; means = np.empty_like(pooled_sum)
    means[present] = pooled_sum[present]/pooled_count[present, None]
    means[~present] = pooled_sum.sum(axis=0)/pooled_count.sum()
    counts = dev_count.sum(axis=1); direction = dev_sum[1]/counts[1]-dev_sum[0]/counts[0]
    composition = (dev_count[1]/counts[1]-dev_count[0]/counts[0])@means
    residual = direction-composition; norm = np.linalg.norm(direction)
    return dict(training_questions=len(training), development_questions=len(development),
        development_steps=int(dev.sum()), unseen_first_token_steps=int(dev_count[:, ~present].sum()),
        composition_projection_fraction=float(composition@direction/(norm*norm)),
        composition_norm_over_full=float(np.linalg.norm(composition)/norm),
        residual_norm_over_full=float(np.linalg.norm(residual)/norm),
        residual_full_cosine=cosine(residual, direction),
        definition='Token means estimated on original400 questions only; decomposition evaluated on original100 development questions. Unseen tokens use pooled training mean. No GPU outcomes or promotion rule involved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable diagnostic exists')
    started = time.perf_counter()
    work = ROOT/'.codex_work/overnight_research_20260912'
    backup = ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    audit = read(work/'lexical_direction_CPU/audit.json')
    for name, digest in audit['source_sha256'].items():
        require(sha(backup/name) == digest, 'Original asset changed: '+name)
    label_path = work/'lexical_direction_CPU/labels.npy'
    require(sha(label_path) == audit['label_array_sha256'], 'Fixed lexical labels changed')
    labels = np.load(label_path) != 0; steps = read(backup/'steps.json')
    question = np.array([s['question'] for s in steps]); first_token = np.empty(len(steps), dtype=np.int64)
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    mapping = []; digest = hashlib.sha256(); capped = 0
    grouped = [[] for _ in range(500)]
    for i, step in enumerate(steps):
        grouped[step['question']].append((i, step))
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as archive:
        for q, line in enumerate(archive.extractfile('generations.jsonl')):
            digest.update(line); row = json.loads(line); source = train[row['train_index']]
            require(row['problem'] == source['problem'], 'Calibration question differs from train')
            mapping.append(dict(train_index=row['train_index']))
            capped += row['finish_reason'] == 'length'
            for i, step in grouped[q]:
                first_token[i] = row['token_ids'][step['start']]
    require(q == 499 and digest.hexdigest() == read(backup/'protocol.json')['source_sha256'], 'Calibration answers changed')
    x = np.load(backup/'layer_21.npy', mmap_mode='r')
    results = {name: decompose(x, labels, values) for name, values in [
        ('first_content_token', first_token),
        ('calibration_question', question)]}
    fit = read(backup/'fit.json'); c = np.array([s['confidence'] for s in steps])
    lexical = np.array([s['lexical_hit'] for s in steps])
    over = lexical | (c < fit['parameters']['q25c'])
    under = ~lexical & (c > fit['parameters']['q75c'])
    original_vector = read_vector(backup/'auto_vector.pt', backup/'auto_vector.pt').astype(np.float64)
    results['original_mixed_direction_first_token'] = decompose(x, over, first_token,
        selected_rows=np.flatnonzero(over | under), reference=original_vector)
    require(results['original_mixed_direction_first_token']['supplied_reference_cosine'] > .999999,
            'Original contrast direction does not reconstruct the frozen vector')
    split = read(backup/'selected_layer.json')
    cross_group = {name: cross_group_decomposition(x, y, first_token, question,
        split['training_questions'], split['validation_questions'], selected_rows=indices)
        for name, y, indices in [('lexical_direction', labels, None),
                               ('original_mixed_direction', over, np.flatnonzero(over | under))]}
    pieces = token_bytes(read(ROOT/'.codex_work/label_audit_30_20260910/tokenizer.json'))
    for item in results['first_content_token']['leave_one_group_out']:
        item['token_text'] = pieces[int(item['group'])].decode('utf-8', errors='replace')
    # Keep all groups in the artifact, without selecting a better vector or gate.
    judgments = audit['joined_content_diagnostics']
    content_counts = {name: dict(Counter(r['content_category'] for r in judgments if r['new_class'] == name))
                      for name in ['execution', 'reflection', 'transition']}
    result = dict(status='descriptive_CPU_lexical_support_audit_no_candidate_change',
        lexical_fit_audit_sha256=sha(work/'lexical_direction_CPU/audit.json'),
        script_sha256=sha(Path(__file__), source=True), generations_sha256=digest.hexdigest(),
        train_sha256=sha(train_path, source=True), questions=500, steps=len(steps), capped_answers_included=capped,
        unavailable_diagnostics='The frozen training JSONL has no type/level metadata; no subject or difficulty subgroup claim is made.',
        decompositions=results, cross_question_group_decompositions=cross_group,
        existing_stratified_manual_cases=content_counts,
        cpu_seconds=time.perf_counter()-started, model_loads=0, GPU_calls=0, new_answers=0,
        decision='Keep the previously fixed vector, scale and GPU gate unchanged; these are descriptive mechanism limitations.',
        limitations=['Grouping does not causally remove token identity or topic from representations.',
            'Step weights match the fixed candidate and long traces retain more influence.',
            'The 76 prior judgments are selected development cases from one reviewer; no population precision or accuracy claim.',
            'No generation, independent correctness validation, classifier training or vector refitting is performed.'])
    save(args.output, result)
    print(json.dumps({name:{k:v for k,v in values.items() if k!='leave_one_group_out'} for name,values in results.items()}))
    print(dict(cpu_seconds=result['cpu_seconds'], manual_cases=content_counts, GPU_calls=0))
    print(json.dumps(cross_group))


if __name__ == '__main__':
    main()
