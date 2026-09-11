"""Fit the prespecified position-only comparison from verified saved states."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from mechanism_candidates import ROOT, BASE, read, save, sha, require, moments, cosine, read_vector, write_vector


def normalize_like(vector, target_norm):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    require(np.isfinite(value).all() and norm > 0, 'Invalid direction')
    out = (value * (target_norm / norm)).astype(np.float32)
    require(abs(float(np.linalg.norm(out.astype(float))) / target_norm - 1) < 1e-6, 'Norm mismatch')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    require(not out.exists(), 'Existing fit is immutable')
    started = time.perf_counter()
    config = read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']
    backup = ROOT/config['inputs']['backup']; ledger = read(a.replay/'ledger.json')
    plan = read(a.prepared/'replay_plan.json')
    require(ledger['status'] == 'completed' and ledger['completed_questions'] == 500, 'Incomplete replay')
    require(ledger['new_answers'] == 0 and ledger['all_original_elements_exact'], 'Saved-state equivalence failed')
    require(sha(a.prepared/'replay_plan.json') == ledger['plan_sha256'], 'Replay plan changed')
    for name, digest in ledger['files_sha256'].items():
        require(sha(a.replay/name) == digest, 'Replay output changed: '+name)
    for name, digest in plan['original_files_sha256'].items():
        require(sha(backup/name) == digest, 'Original input changed: '+name)
    require(sha(a.prepared/'matched_mask.npy') == plan['matched_mask_sha256'], 'Support changed')
    steps = read(backup/'steps.json'); protocol = read(backup/'protocol.json')
    selection = read(backup/'selected_layer.json')
    train, valid = [set(selection[key]) for key in ('training_questions', 'validation_questions')]
    require(len(train) == 400 and len(valid) == 100 and not train & valid, 'Question split changed')
    q = np.array([s['question'] for s in steps]); c = np.array([s['confidence'] for s in steps])
    lex = np.array([s['lexical_hit'] for s in steps]); mask = np.load(a.prepared/'matched_mask.npy')
    low, high = protocol['confidence_quantiles']
    over, under = mask & (lex | (c < low)), mask & ~lex & (c > high)
    masks = {'over':over, 'under':under}
    for split, questions in [('train',train), ('valid',valid)]:
        keep = np.isin(q, list(questions))
        for name, selected in [('over',over), ('under',under)]: masks[split+'_'+name] = keep & selected
    original = read_vector(backup/'auto_vector.pt', backup/'auto_vector.pt')
    target_norm = float(np.linalg.norm(original.astype(float)))
    states = {'matched_first':np.load(backup/'layer_21.npy', mmap_mode='r'),
              'predecessor_delimiter':np.load(a.replay/'predecessor_layer21.npy', mmap_mode='r')}
    vectors, result = {}, {}
    class_questions = {name:int(len(np.unique(q[selected]))) for name, selected in masks.items()}
    for name, x in states.items():
        require(x.shape == (84008,1536) and x.dtype == np.float32, 'Wrong state layout')
        stats = moments(x, masks)
        direction = stats['over']['mean'] - stats['under']['mean']
        d_train = stats['train_over']['mean'] - stats['train_under']['mean']
        d_valid = stats['valid_over']['mean'] - stats['valid_under']['mean']
        vectors[name] = normalize_like(direction, target_norm)
        result[name] = dict(raw_norm=float(np.linalg.norm(direction)),
            saved_norm=float(np.linalg.norm(vectors[name].astype(float))),
            train_validation_cosine=cosine(d_train,d_valid), original_cosine=cosine(direction,original))
    difference = float(np.linalg.norm(vectors['predecessor_delimiter'].astype(float)-vectors['matched_first'])/target_norm)
    gate = config['cpu_vector_gate']
    passes = (min(class_questions.values()) >= gate['minimum_class_questions']
        and result['predecessor_delimiter']['train_validation_cosine'] >= gate['minimum_train_validation_direction_cosine']
        and difference >= gate['minimum_relative_change_from_matched_first'])
    out.mkdir(parents=True)
    for name, vector in vectors.items():
        folder = out/name; folder.mkdir()
        write_vector(folder/'auto_vector.pt', vector, backup/'auto_vector.pt')
        fit = read(backup/'fit.json')
        fit.update(version='position-only-v1', vector_sha256=sha(folder/'auto_vector.pt'),
            vector_norm=float(np.linalg.norm(vector.astype(float))), positives=int(over.sum()), negatives=int(under.sum()),
            method='Original mixed labels and fixed original dynamic controller; generated-predecessor support only; '+name+' states; same original vector norm. No new answers, no layer or parameter search.',
            original_fit_sha256=sha(backup/'fit.json'), replay_ledger_sha256=sha(a.replay/'ledger.json'))
        save(folder/'fit.json', fit)
        result[name].update(vector_sha256=sha(folder/'auto_vector.pt'), fit_sha256=sha(folder/'fit.json'))
    summary = dict(status='ready_for_fixed_gpu_screen' if passes else 'stopped_cpu_gate',
        vectors=result, class_questions=class_questions, class_steps={name:int(mask.sum()) for name,mask in masks.items()},
        relative_position_change=difference, position_cosine=cosine(vectors['matched_first'],vectors['predecessor_delimiter']),
        gate=gate, passes_cpu_gate=passes, seconds=time.perf_counter()-started,
        replay_ledger=ledger, source_sha256=sha(Path(__file__),source=True),
        prepared_plan_sha256=sha(a.prepared/'replay_plan.json'),
        limitation='The 400/100 groups are original calibration development groups already used for layer selection. Direction stability and position difference are mechanism diagnostics, not independent confirmation or generation gains.')
    save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True))


if __name__ == '__main__': main()
