"""CPU projection of the frozen controller onto saved, unsteered training traces.

Joins existing content evidence to the preceding and following boundaries. This
is neither an actual steered rollout nor an estimate of intervention benefit.
Uses NumPy to execute the controller's small numerical functions without torch.
"""
import argparse
import ast
from collections import Counter
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np

from audit_calibration_labels import ROOT, DEFAULT, read, save, sha

CONFIG = ROOT / 'integration/rebalance_easysteer/configs'
RUNTIME = ROOT / 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
OUTPUT = ROOT / '.codex_work/control_alignment30_20260910'
EPS = 1e-6


class Array(np.ndarray):
    """Only the tensor operations needed by the extracted controller functions."""
    @property
    def device(self):
        return 'cpu'

    def clamp(self, min=None, max=None):
        return np.asarray(np.clip(self, min, max)).view(Array)


def tensor(value, device='cpu', dtype=np.float64):
    assert device == 'cpu'
    return np.asarray(value, dtype=dtype).view(Array)


def load_controller():
    names = {'_solve_k_for_tau', 'validate_curve_targets', '_baseline',
             '_curve_constants', 'compute_rebalance_coefficient'}
    nodes = [n for n in ast.parse(RUNTIME.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    backend = SimpleNamespace(Tensor=Array, tensor=tensor, tanh=np.tanh,
        nan_to_num=lambda x, **kw: tensor(np.nan_to_num(x, **kw), dtype=x.dtype),
        sigmoid=lambda x: 1 / (1 + np.exp(-x)),
        no_grad=nullcontext)
    ns = dict(torch=backend, math=math, lru_cache=lru_cache,
              ReBalanceParams=SimpleNamespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNTIME), 'exec'), ns)
    return ns


def scalar_reference(c, v, p, constants):
    """Independent scalar decomposition to check the extracted array execution."""
    mid, k, intercept, slope, at_q25, at_one = constants
    sig = lambda x: 1 / (1 + math.exp(-min(60., max(-60., x))))
    base = intercept + slope * math.tanh(k * (c - mid))
    dc, dv = max(1e-12, p.q75c - p.q25c), max(1e-12, p.q75v - p.q25v)
    lw = min(1., sig((p.q25c-c)/dc*1200) * sig((v-p.q75v)/dv*1200) / .25)
    normalizer = sig((1-p.q75c)/dc*12) * .5
    hw = min(1., sig((c-p.q75c)/dc*1200) * sig((p.q25v-v)/dv*1200) / normalizer)
    low, high = (p.low_val_2-at_q25)*lw, (p.high_val_2-at_one)*hw
    value = min(max(.01, p.high_val_2), max(min(p.low_val_1, p.low_val_2), base+low+high))
    return dict(base=float(base), low_gate_weight=float(lw), high_gate_weight=float(hw),
                low_adjustment=float(low), high_adjustment=float(high), coefficient=float(value))


def sign(value):
    return 'positive' if value > EPS else 'negative' if value < -EPS else 'near_zero'


def boundary_view(step):
    if step is None:
        return dict(status='no_initial_prompt_injection', coefficient=None)
    if not step['has_boundary']:
        return dict(status='think_end_not_an_injection_boundary', coefficient=None)
    return {k: step[k] for k in ['step', 'confidence', 'variance', 'coefficient',
        'coefficient_float32_emulation', 'base', 'low_gate_weight', 'high_gate_weight',
        'low_adjustment', 'high_adjustment', 'boundary_token_offset']} | dict(status='projected_boundary')


def join_point(point, steps):
    n = point['step']
    return dict(**{k: point[k] for k in ['question', 'train_index', 'case_id', 'step',
        'progress_evidence', 'math_validity', 'progress_quote', 'answer_evidence',
        'latest_reviewed_answer_step', 'latest_reviewed_answer_verdict']},
        incoming=boundary_view(steps[n-2] if n > 1 else None),
        outgoing=boundary_view(steps[n-1]))


def summarize(points, side, moderate):
    groups = {}
    for category in ['new', 'repeat', 'uncertain', 'new_supported']:
        rows = [p for p in points if (p['progress_evidence'] == category or
            category == 'new_supported' and p['progress_evidence'] == 'new' and p['math_validity'] == 'supported')]
        active = [p[side]['coefficient'] for p in rows if p[side]['coefficient'] is not None]
        groups[category] = dict(total=len(rows), projected_boundaries=len(active),
            excluded=len(rows)-len(active), signs=dict(Counter(map(sign, active))),
            negative_at_least_moderate=sum(v <= moderate + EPS for v in active),
            coefficient_min=min(active, default=None), coefficient_max=max(active, default=None))
    return groups


def run(output, report_path):
    output.mkdir(parents=True, exist_ok=True)
    if report_path.exists() or (output / 'projected_steps.json').exists():
        raise FileExistsError('Choose a new output; existing audit artifacts are immutable')
    frozen_path = CONFIG / 'final_results_20260909.json'
    fit_path = CONFIG / 'auto_code_v2_1p5b_20260908.json'
    evidence_path = CONFIG / 'calibration_label_evidence30_20260910.json'
    frozen, fit, evidence = read(frozen_path), read(fit_path), read(evidence_path)
    protocol, key = read(DEFAULT / 'audit_protocol.json'), read(DEFAULT / 'private_key.json')
    assert sha(DEFAULT / 'private_key.json') == protocol['private_key_sha256']
    index_path = ROOT / evidence['artifacts']['index_path']
    assert sha(index_path) == evidence['artifacts']['index_sha256']
    index = read(index_path)
    expected_source = fit['calibration_source_sha256']
    assert expected_source == evidence['protocol']['source_sha256'] == protocol['source_sha256']
    source_git = subprocess.check_output(['git', 'show', 'HEAD:' + RUNTIME.relative_to(ROOT).as_posix()], cwd=ROOT)
    assert hashlib.sha256(source_git).hexdigest() == frozen['source_sha256'][RUNTIME.relative_to(ROOT).as_posix()]
    assert source_git.decode().replace('\r\n', '\n') == RUNTIME.read_text(encoding='utf-8')
    p = SimpleNamespace(**fit['parameters'])
    dynamic = frozen['benchmarks'][0]['protocol']['dynamic_params']
    assert all(dynamic[k] == v for k, v in fit['parameters'].items())
    boundaries = set(dynamic['boundary_token_ids'])
    runtime = load_controller()
    constants = runtime['_curve_constants'](p.q25c, p.q75c, p.low_val_1, 'cpu', np.dtype('float64'), p.curve_tau)
    historical_curve = read(ROOT / '.codex_work/auto_code_v2_500_20260908/curve_check.json')
    actual_anchors = runtime['_baseline'](tensor([p.q25c, p.q75c, 1.]), *constants[:4])
    np.testing.assert_allclose(actual_anchors, historical_curve['anchors'], atol=1e-12, rtol=0)
    assert abs(constants[1] - historical_curve['k']) < 1e-12
    selected = {q['calibration_index']: q for q in index['questions']}
    all_questions = []
    with tarfile.open(ROOT / protocol['source_archive']) as archive:
        with archive.extractfile(protocol['source_member']) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_source
        import json
        for i, line in enumerate(archive.extractfile(protocol['source_member'])):
            if i not in selected:
                continue
            raw, saved = json.loads(line), selected[i]
            assert raw['train_index'] == saved['train_index'] and raw['problem'] == saved['problem']
            assert len(raw['token_ids']) == saved['output_tokens'] <= 16000
            previous, previous32, steps = None, None, []
            for s in saved['steps']:
                a, b = s['start'], s['stop']
                probs = [math.exp(x) for x in raw['logprobs'][a:b]]
                assert probs and all(0 <= x <= 1.00001 for x in probs)
                c = math.fsum(probs)/len(probs)
                v = 0. if previous is None else (c-previous)**2/4
                c32 = np.cumsum(np.asarray(probs, dtype=np.float32), dtype=np.float32)[-1]/np.float32(len(probs))
                v32 = np.float32(0) if previous32 is None else np.square(c32-previous32)/np.float32(4)
                has_boundary = b < len(raw['token_ids']) and raw['token_ids'][b] in boundaries
                assert has_boundary or raw['token_ids'][b] == dynamic['think_end_token_id']
                steps.append(dict(step=s['step'], start=a, stop=b, confidence=c, variance=v,
                    confidence_float32=float(c32), variance_float32=float(v32),
                    has_boundary=has_boundary, boundary_token_offset=b if has_boundary else None))
                previous, previous32 = c, c32
            all_questions.append(dict(question=saved['question'], train_index=saved['train_index'],
                calibration_index=i, finish_reason=saved['finish_reason'], steps=steps))
    flat = [s for q in all_questions for s in q['steps']]
    coefficients = runtime['compute_rebalance_coefficient'](tensor([s['confidence'] for s in flat]), tensor([s['variance'] for s in flat]), p)
    coefficients32 = runtime['compute_rebalance_coefficient'](tensor([s['confidence_float32'] for s in flat], dtype=np.float32), tensor([s['variance_float32'] for s in flat], dtype=np.float32), p)
    errors = []
    for s, actual, approximate32 in zip(flat, coefficients, coefficients32):
        details = scalar_reference(s['confidence'], s['variance'], p, constants)
        errors.append(abs(details['coefficient'] - actual))
        s.update(details, coefficient_float32_emulation=float(approximate32))
    assert len(flat) == 4652 and max(errors) < 1e-12
    by_q = {q['question']: q for q in all_questions}
    for k in key:
        s = by_q[k['question']]['steps'][k['step']-1]
        assert s['start'] == k['start'] and s['stop'] == k['stop']
        assert abs(s['confidence']-k['confidence']) < 1e-12 and abs(s['variance']-k['variance']) < 1e-12
    points, anchors = [], []
    labels = {k['case_id']: k for k in key}
    for q in evidence['questions']:
        steps = by_q[q['question']]['steps']
        for point in q['checkpoints']:
            joined = join_point(point, steps)
            old = labels[point['case_id']]
            joined.update(original_label=old['original_label'], original_label_route=old['route'])
            points.append(joined)
        for anchor in q['answer_anchors']:
            anchors.append(dict(question=q['question'], train_index=q['train_index'],
                **{k: anchor[k] for k in ['step', 'scope', 'answer', 'verdict', 'quote']},
                incoming=boundary_view(steps[anchor['step']-2] if anchor['step'] > 1 else None),
                outgoing=boundary_view(steps[anchor['step']-1])))
    assert len(points) == 76 and len(anchors) == 44
    save(output / 'projected_steps.json', all_questions)
    result = dict(status='completed_cpu_controller_projection_not_intervention_test',
        purpose='Align existing content evidence with before/after boundary coefficients on saved unsteered training traces',
        execution_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        question_count=30, complete_steps=len(flat), reviewed_checkpoints=len(points), answer_witnesses=len(anchors),
        capped_questions=sum(q['finish_reason']=='length' for q in all_questions),
        new_generations=0, model_forwards=0, gpu_runs=0, parameters=fit['parameters'],
        timing='incoming uses only the preceding completed step; outgoing follows the reviewed step. First prompt and think_end have no boundary injection.',
        coefficient_interpretation='Counterfactual formula projection on unsteered prefixes, not the coefficients observed in a steered rollout; float64 NumPy execution of extracted frozen source.',
        sign_zero_tolerance=EPS, moderate_cutoff=p.low_val_1,
        checks=dict(saved_confidence_variance_matches=76, historical_curve_anchors_max_error=float(np.max(np.abs(actual_anchors-historical_curve['anchors']))),
            scalar_array_max_error=max(errors), float32_emulation_max_coefficient_difference=float(np.max(np.abs(coefficients-coefficients32))),
            float32_emulation_sign_disagreements=sum(sign(a)!=sign(b) for a,b in zip(coefficients,coefficients32)),
            float32_limit='NumPy sequential probability accumulation and controller arithmetic; no claim of CUDA softmax/kernel rounding equivalence'),
        summaries={side: summarize(points, side, p.low_val_1) for side in ['incoming', 'outgoing']},
        limitations=['Same 30 exposed development questions; original strata uneven; no population prevalence or held-out accuracy.',
            'New/repeat/uncertain are single-reviewer content categories, not over/under/normal ground truth.',
            'A coefficient sign is an intended direction, not proof of actual length or correctness change.',
            'Projected previous-boundary coefficients cannot establish that a generated step would survive actual intervention.',
            'Correct witnesses do not prove reliable answerability; absence of a witness is not underthinking.',
            'No threshold fitting, label replacement, vector extraction, model run, or frozen-result overwrite.'],
        artifacts=dict(script_sha256=sha(__file__), runtime_source_sha256=sha(RUNTIME), runtime_git_content_sha256=hashlib.sha256(source_git).hexdigest(),
            fit_sha256=sha(fit_path), frozen_manifest_sha256=sha(frozen_path), evidence_sha256=sha(evidence_path),
            source_generations_sha256=expected_source, projected_steps_path=str((output/'projected_steps.json').relative_to(ROOT)),
            projected_steps_sha256=sha(output/'projected_steps.json')),
        checkpoints=points, answer_anchors=anchors)
    save(report_path, result)
    import json
    print(json.dumps({k: result[k] for k in ['question_count','complete_steps','reviewed_checkpoints','checks','summaries']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.report)
