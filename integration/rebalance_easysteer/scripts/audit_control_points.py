"""Prepare a position-alignment replay from saved calibration answers, on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from mechanism_candidates import ROOT, BASE, read, save, sha, require, moments, read_vector, cosine


def map_positions(row, steps, saved, boundary):
    """Map each nonempty complete step to its content and predecessor input."""
    prompt = len(row['prompt_token_ids'])
    ids = row['token_ids']
    first, predecessor, generated = [], [], []
    for s in steps:
        require(0 <= s['start'] < s['stop'] <= len(ids), 'Invalid content interval')
        require(ids[s['start']] not in boundary, 'First content is a delimiter')
        p = prompt + s['start']
        valid = s['start'] > 0
        if valid:
            require(ids[s['start'] - 1] in boundary, 'Predecessor is not a delimiter')
        first.append(p)
        predecessor.append(p - 1)
        generated.append(valid)
    require(first == saved['positions'], 'Original feature positions differ')
    require(prompt > 0 and all(x >= 0 for x in predecessor), 'Missing prompt predecessor')
    return dict(question=saved['question'], first=first, predecessor=predecessor,
                generated_predecessor=generated, prompt_tokens=prompt,
                think_stop=saved['think_stop'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    require(not out.exists(), 'Audit already exists')
    config = read(ROOT / BASE / 'configs/overnight_research_20260912.json')
    inputs = config['first_investigation']['inputs']
    backup = ROOT / inputs['backup']; steps = read(backup / 'steps.json')
    positions = read(backup / 'positions.json'); protocol = read(backup / 'protocol.json')
    tokenizer = read(ROOT / inputs['tokenizer'])
    vocab = tokenizer['model']['vocab']; boundary = {i for t, i in vocab.items() if 'ĊĊ' in t}
    by_id = {i: t for t, i in vocab.items()}
    started = time.perf_counter()
    with tarfile.open(ROOT / inputs['archive']) as archive:
        raw = archive.extractfile(inputs['member']).read()
    require(hashlib.sha256(raw).hexdigest() == inputs['generations_sha256'], 'Saved answers changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == len(positions) == 500 and len(steps) == 84008, 'Wrong original data')
    groups = [[] for _ in rows]
    for s in steps: groups[s['question']].append(s)
    maps = [map_positions(r, group, pos, boundary) for r, group, pos in zip(rows, groups, positions)]
    valid = np.array([v for m in maps for v in m['generated_predecessor']])
    c = np.array([s['confidence'] for s in steps]); lex = np.array([s['lexical_hit'] for s in steps])
    low, high = protocol['confidence_quantiles']
    over, under = lex | (c < low), ~lex & (c > high)
    masks = dict(over=over, under=under, matched_over=over & valid, matched_under=under & valid)
    x = np.load(backup / 'layer_21.npy', mmap_mode='r')
    stats = moments(x, masks)
    d = stats['over']['mean'] - stats['under']['mean']
    old = read_vector(backup / 'auto_vector.pt', backup / 'auto_vector.pt')
    require(np.array_equal(d.astype(np.float32), old), 'Original vector did not reproduce')
    matched = stats['matched_over']['mean'] - stats['matched_under']['mean']
    q = np.array([s['question'] for s in steps])
    first_tokens = np.array([rows[s['question']]['token_ids'][s['start']] for s in steps])
    frequent = {}
    for name, mask in masks.items():
        frequent[name] = [{'token_id': int(i), 'token': by_id.get(int(i), '<special>'), 'count': int(n)}
                          for i, n in Counter(first_tokens[mask]).most_common(12)]
    out.mkdir(parents=True)
    save(out / 'positions.json', maps)
    np.save(out / 'matched_mask.npy', valid)
    summary = dict(status='cpu_position_audit_completed', count=500, steps=len(steps),
        first_content_positions=len(steps), same_token_as_injected_predecessor=0,
        valid_generated_predecessors=int(valid.sum()), excluded_prompt_predecessors=int((~valid).sum()),
        class_counts={k: int(mask.sum()) for k, mask in masks.items()},
        class_questions={k: int(len(np.unique(q[mask]))) for k, mask in masks.items()},
        original_vector_bitwise_reconstructed=True,
        matched_first_vs_original_cosine=cosine(matched, d),
        original_norm=float(np.linalg.norm(d)), matched_first_norm=float(np.linalg.norm(matched)),
        first_token_class_frequencies=frequent, seconds=time.perf_counter() - started,
        new_generations=0, model_forwards=0,
        limitation='Position mismatch is confirmed; no causal generation benefit established. Labels contain future step information only offline. Prompt predecessors are excluded from matched fitting.',
        source_sha256={str(Path(BASE) / 'scripts/audit_control_points.py'): sha(Path(__file__), source=True)},
        inputs_sha256={n: sha(backup/n) for n in ['steps.json','positions.json','protocol.json','layer_21.npy','fit.json','auto_vector.pt']})
    save(out / 'summary.json', summary)
    model_hash = read(ROOT / BASE / 'configs/first_step_validation100_20260910.json')['model_verified_against_freeze']
    replay = dict(status='prepared_replay_saved_answers_only', questions=500, steps=len(steps), width=1536,
        decoder_output_layer=20, model=protocol['model'], model_files_sha256=model_hash,
        generations=str(Path(protocol['source']) / 'generations.jsonl').replace('\\','/'),
        generations_sha256=inputs['generations_sha256'],
        original='/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908',
        original_files_sha256=summary['inputs_sha256'], mapping_sha256=sha(out/'positions.json'),
        matched_mask_sha256=sha(out/'matched_mask.npy'), cpu_summary_sha256=sha(out/'summary.json'),
        source_plan_sha256=sha(ROOT / BASE / 'configs/overnight_research_20260912.json',source=True),
        method='Causal teacher-forced decoder replay of saved tokens; predecessor input states at original fixed layer. No model.generate or new answers.',
        capture=['predecessor states for all complete steps, with generated-only mask','mean prompt states at the same layer for possible later nuisance analysis'],
        original_feature_check='Compare all first-content states in the same forward pass; report exact elements, relative RMSE and cosine. Stop if per-question relative RMSE exceeds 0.02; this tolerates BF16/runtime numerical differences without claiming bitwise equivalence.',
        timeout_seconds=900, expected_new_bytes=len(steps)*1536*4+500*1536*4+5000000,
        output='/root/autodl-tmp/results/easysteer/control_point_replay_20260912',
        no_generation_benefit_claim=True)
    save(out/'replay_plan.json',replay)
    print(json.dumps({k:v for k,v in summary.items() if k!='first_token_class_frequencies'},ensure_ascii=True))


if __name__ == '__main__': main()
