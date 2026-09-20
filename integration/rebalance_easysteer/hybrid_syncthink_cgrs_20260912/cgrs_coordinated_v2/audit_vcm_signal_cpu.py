"""Audit a proposed VCM-scale substitution on saved logits, without generation.

Tail perturbations are counterexamples, not estimates of natural model drift.
Only the reviewed upstream class is evaluated, with its HF base stubbed out.
"""
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'vcm_signal_audit_20260918'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(x):
    p = torch.softmax(x.double(), -1)
    finite = torch.isfinite(x)
    return dict(pmax=float(p.max()), entropy=float(-(p * p.clamp_min(1e-300).log()).sum()),
                sigma_finite=float(x[finite].double().std(unbiased=False)),
                finite_count=int(finite.sum()), vocabulary_width=x.numel())


def tests(upstream):
    tree = ast.parse(upstream.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VCMLogitsProcessor')
    ns = {'torch': torch, 'LogitsProcessor': object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(upstream), 'exec'), ns)
    processor = ns['VCMLogitsProcessor']
    z = torch.tensor([[2., 1., 0., -1.]], dtype=torch.float64)
    prior = torch.tensor([0., .2, .3, .1], dtype=z.dtype)
    ids = torch.tensor([[1, 1, 2]])
    p = processor(prior, alpha=.1)
    got = p(ids, z)
    counts = torch.tensor([[0., 2., 1., 0.]], dtype=z.dtype)
    expected = z + .1 * (z-prior-counts*z.std(dim=-1, keepdim=True))
    assert torch.equal(got, expected)
    assert torch.equal(processor(prior, alpha=0)(ids, z), z)
    masked = z.clone(); masked[0, -1] = -torch.inf
    nonfinite = processor(prior, alpha=.1)(ids, masked)
    assert torch.isnan(nonfinite[0, 1])
    # Nearly identical probabilities but a different low-probability tail scale.
    a = torch.tensor([0., -1.] + [-40.]*998, dtype=torch.float64)
    b = a.clone(); b[501:] -= 100.
    tv = float((a.softmax(-1)-b.softmax(-1)).abs().sum()/2)
    assert tv < 1e-12 and b.std(unbiased=False) > 10*a.std(unbiased=False)
    # Large global spread is not a mathematical guarantee of high pmax.
    certain = torch.tensor([8.]+[0.]*999, dtype=torch.float64)
    uncertain = torch.tensor([0., 0.]+[-100.]*998, dtype=torch.float64)
    assert moments(uncertain)['sigma_finite'] > moments(certain)['sigma_finite']
    assert moments(uncertain)['pmax'] < moments(certain)['pmax']
    # A small alpha cannot universally make distinct priors indistinguishable.
    neutral = torch.zeros((1, 2), dtype=torch.float64)
    first = processor(torch.tensor([0., 0.]), alpha=.1, mode='pmi_only')(ids[:, :0], neutral)
    second = processor(torch.tensor([10., 0.]), alpha=.1, mode='pmi_only')(ids[:, :0], neutral)
    return dict(checks_passed=6, upstream_formula='sample std correction=1; paper Eq5 population correction=0',
                masked_logit_produces_nan=True,
                tail_counterexample=dict(original=moments(a), perturbed=moments(b), probability_tv=tv),
                confidence_order_counterexample=dict(certain=moments(certain), uncertain=moments(uncertain)),
                prior_counterexample_probability_tv=float((first.softmax(-1)-second.softmax(-1)).abs().sum()/2))


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    plan_path = OUT/'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    upstream = ROOT/'.codex_work/vcm_audit_20260918/vcm_processors.py'
    assert sha(upstream) == plan['upstream_source_sha256']
    checks = tests(upstream)
    src = Path(plan['capture_root'])
    manifest = json.loads((src/'manifest.json').read_text())
    mapping = json.loads((src/'results/runtime.json').read_text())['request_mapping']
    registry = {r['train_index']:r for r in json.loads((HERE/'same_logits8_20260916/plan.json').read_text())['rows']}
    seen, rows, files = set(), [], {}
    for path in sorted((src/'results').glob('capture_*.pt')):
        assert sha(path) == manifest['results/'+path.name]
        files[path.name] = sha(path)
        blob = torch.load(path, map_location='cpu', weights_only=True)
        for j,(rid,slot,valid) in enumerate(zip(blob['requests'],blob['slots'],blob['valid'])):
            if not valid: continue
            count = int(blob['owner']['count'][slot]); key = (rid,count)
            if key in seen: continue
            seen.add(key)
            x = blob['logits'][j].double()
            finite = torch.isfinite(x)
            assert not torch.isnan(x).any() and not torch.isposinf(x).any() and finite.any()
            base = moments(x)
            # Fixed stress: subtract 32 from bottom half of finite vocabulary.
            indices = torch.nonzero(finite).flatten()
            bottom = indices[torch.argsort(x[indices])[:len(indices)//2]]
            changed = x.clone(); changed[bottom] -= plan['tail_logit_subtraction']
            after = moments(changed)
            p,q = x.softmax(-1),changed.softmax(-1)
            coef = float(blob['coefs'][slot]); mean = float(blob['prev_step_mean'][slot])
            eligible = bool(blob['owner']['opening'][slot] and blob['owner']['thinking'][slot]
                            and math.isfinite(coef) and math.isfinite(mean) and coef < 0)
            rows.append(dict(train_index=mapping[rid], problem_sha256=registry[mapping[rid]]['problem_sha256'],
                generated_position=count, original_rc14_eligible=eligible, original=base,
                tail_stress=after, original_tail_mass=float(p[bottom].sum()),
                probability_tv=float((p-q).abs().sum()/2),
                sigma_ratio=after['sigma_finite']/base['sigma_finite']))
    sigma=np.array([r['original']['sigma_finite'] for r in rows])
    pmax=np.array([r['original']['pmax'] for r in rows])
    report=dict(status='CPU audit complete; no candidate efficacy run justified by these captures',
        plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),upstream_sha256=sha(upstream),
        model='DeepSeek-R1-Distill-Qwen-7B',capture_batches=len(files),positions=len(rows),
        questions=len({r['train_index'] for r in rows}),
        original_rc14_eligible_positions=sum(r['original_rc14_eligible'] for r in rows),
        nonfinite_rows=sum(r['original']['finite_count']!=r['original']['vocabulary_width'] for r in rows),
        descriptive_pearson_sigma_pmax=float(np.corrcoef(sigma,pmax)[0,1]),
        sigma_range=[float(sigma.min()),float(sigma.max())],
        tail_stress_max_probability_tv=max(r['probability_tv'] for r in rows),
        tail_stress_median_sigma_ratio=float(np.median([r['sigma_ratio'] for r in rows])),
        checks=checks,rows=rows,input_sha256=files,
        new_model_forwards=0,new_answers=0,cpu_seconds=time.monotonic()-started,
        decision='Do not directly replace L27 strength with global sigma as a confidence score. No statement that full VCM is ineffective.',
        limitations=plan['limitations'])
    with (OUT/'result.json').open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','input_sha256')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
