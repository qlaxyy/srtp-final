"""Freeze 20 training prefixes for one-boundary apply/skip continuations on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np

from audit_calibration_labels import ROOT, DEFAULT, read, save, sha
from audit_control_alignment import CONFIG, RUNTIME, load_controller, tensor, EPS


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def completed_boundaries(ids, logs, boundaries, end_id):
    """Only past probabilities determine the state; skip empty/terminal spans."""
    previous, previous32, start, step = None, None, 0, 0
    for stop, token in enumerate(ids):
        if token == end_id:
            break
        if token not in boundaries:
            continue
        if stop > start:
            probs = [math.exp(p) for p in logs[start:stop]]
            c = math.fsum(probs) / len(probs)
            c32 = np.cumsum(np.asarray(probs, dtype=np.float32), dtype=np.float32)[-1] / np.float32(len(probs))
            v = 0. if previous is None else (c-previous)**2 / 4
            v32 = np.float32(0) if previous32 is None else np.square(c32-previous32) / np.float32(4)
            step += 1
            if stop + 1 < 16000:
                yield dict(step=step, boundary_output_offset=stop,
                    confidence=c, variance=v, confidence_float32=float(c32),
                    variance_float32=float(v32), step_tokens=stop-start)
            previous, previous32 = c, c32
        start = stop + 1


def select_cases(candidates, excluded, seed=20260910, per_stratum=10):
    rng = random.Random(seed)
    selected, used, availability = [], set(excluded), {}
    for label in ('strong_negative', 'positive'):
        pool = sorted(q for q, groups in candidates.items() if q not in used and groups[label])
        availability[label] = len(pool)
        for q in sorted(rng.sample(pool, min(per_stratum, len(pool)))):
            chosen = rng.choice(candidates[q][label])
            selected.append(dict(calibration_index=q, stratum=label, **chosen))
            used.add(q)
    return selected, availability


def prepare(output):
    # Directory itself is reserved to prevent accidental resampling/overwrites.
    output.mkdir(parents=True, exist_ok=False)
    audit = read(DEFAULT / 'audit_protocol.json')
    frozen = read(CONFIG / 'final_results_20260909.json')
    fit_path = CONFIG / 'auto_code_v2_1p5b_20260908.json'
    fit = read(fit_path)
    params = frozen['benchmarks'][0]['protocol']['dynamic_params']
    assert all(params[k] == v for k, v in fit['parameters'].items())
    source = subprocess.check_output(['git', 'show', 'HEAD:' + RUNTIME.relative_to(ROOT).as_posix()], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == frozen['source_sha256'][RUNTIME.relative_to(ROOT).as_posix()]
    assert RUNTIME.read_text(encoding='utf-8') == source.decode().replace('\r\n', '\n')
    boundaries, end_id = set(params['boundary_token_ids']), params['think_end_token_id']
    excluded = audit['selected_calibration_indices']
    runtime, p = load_controller(), SimpleNamespace(**fit['parameters'])
    archive_path = ROOT / Path(audit['source_archive'].replace('\\', '/'))
    rows, candidates = [], {}
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        assert manifest['count'] == 500 and manifest['temperature'] == 0 and manifest['max_tokens'] == 16000
        with archive.extractfile(audit['source_member']) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == fit['calibration_source_sha256'] == audit['source_sha256']
        for q, line in enumerate(archive.extractfile(audit['source_member'])):
            row = json.loads(line)
            rows.append(row)
            assert row['train_index'] == manifest['train_indices'][q]
            if q in excluded:
                continue
            steps = list(completed_boundaries(row['token_ids'], row['logprobs'], boundaries, end_id))
            coefs = runtime['compute_rebalance_coefficient'](tensor([s['confidence'] for s in steps]), tensor([s['variance'] for s in steps]), p)
            coefs32 = runtime['compute_rebalance_coefficient'](tensor([s['confidence_float32'] for s in steps], dtype=np.float32), tensor([s['variance_float32'] for s in steps], dtype=np.float32), p)
            candidates[q] = dict(strong_negative=[], positive=[])
            for s, coef, coef32 in zip(steps, coefs, coefs32, strict=True):
                s.update(projected_coefficient=float(coef), projected_coefficient_float32=float(coef32))
                if coef <= p.low_val_1 + EPS:
                    candidates[q]['strong_negative'].append(s)
                elif coef > EPS:
                    candidates[q]['positive'].append(s)
    assert len(rows) == 500
    selected, availability = select_cases(candidates, excluded)
    train = [json.loads(line) for line in (ROOT / 'sources/ReBalance/Data/Math_Train/test.jsonl').read_text(encoding='utf-8').splitlines() if line]
    tests = set()
    for name in ('Math_Math500', 'Math_GSM8K'):
        tests.update(''.join(json.loads(line)['problem'].split()) for line in
            (ROOT / 'sources/ReBalance/Data' / name / 'test.jsonl').read_text(encoding='utf-8').splitlines() if line)
    inputs = []
    for i, case in enumerate(selected):
        row = rows[case['calibration_index']]
        assert row['problem'] == train[row['train_index']]['problem']
        assert ''.join(row['problem'].split()) not in tests
        n = case['boundary_output_offset'] + 1
        prefix = row['token_ids'][:n]
        prompt = row['prompt_token_ids'] + prefix
        assert prefix[-1] in boundaries and end_id not in prompt and params['think_start_token_id'] in prompt
        case.update(case_id=f'B{i+1:02d}', train_index=row['train_index'],
            prefix_generated_tokens=n, original_prompt_tokens=len(row['prompt_token_ids']),
            remaining_tokens=16000-n, prefix_sha256=ids_hash(prefix),
            prompt_sha256=ids_hash(prompt), source_output_tokens=len(row['token_ids']),
            source_finish_reason=row['finish_reason'])
        # No reference answer or future generated token is sent to the model.
        inputs.append(dict(case_id=case['case_id'], prompt_token_ids=prompt,
            generated_prefix_token_ids=prefix, problem=row['problem']))
    save(output / 'model_inputs.json', inputs)
    result = dict(status='prepared_not_generated', purpose='One-boundary apply/skip diagnostic; no strength search',
        selection_seed=20260910, generation_seed=42, temperature=0, top_p=.95,
        generated_token_budget_including_prefix=16000, question_count=len(selected),
        planned_continuations=2*len(selected), excluded_calibration_indices=excluded,
        sampling='Question first, strong-negative stratum first, then positive among remaining questions; uniform eligible boundary within each selected question',
        interpretation='Previously unreviewed calibration questions, not independent validation. No future correctness/text used to select boundaries.',
        availability_questions=availability, cases=selected, parameters=params,
        model=manifest['model'], decoder_output_layer=fit['decoder_output_layer'],
        new_generations=0, model_forwards=0, preparation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        artifacts=dict(source_generations_sha256=audit['source_sha256'], fit_sha256=sha(fit_path),
            vector_sha256=fit['vector_sha256'], frozen_manifest_sha256=sha(CONFIG/'final_results_20260909.json'),
            controller_git_content_sha256=hashlib.sha256(source).hexdigest(),
            train_sha256=manifest['train_sha256'], tokenizer_sha256=audit['tokenizer_sha256'],
            script_sha256=sha(__file__), model_inputs_sha256=sha(output/'model_inputs.json')))
    save(output / 'manifest.json', result)
    print(json.dumps({k:result[k] for k in ('status','question_count','planned_continuations','availability_questions')}, indent=2))
    print('prefix generated token lengths:', [s['prefix_generated_tokens'] for s in selected])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
