"""CPU-only label ablation using immutable saved calibration trajectories.

No model loading, answer generation, layer reselection, or benchmark selection.
The lexical online controller below is a new hypothesis, not author code.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from policy import TRIGGERS
from review_wsc_native_cpu import Decoder


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def labels(c, v, lexical, params, rule):
    if rule == 'lexical':
        return lexical | (c < params['q25c']), ~lexical & (c > params['q75c'])
    if rule == 'confidence_variance':
        # Inclusive comparisons follow Eq. 5; arithmetic confidence is retained.
        return ((c <= params['q25c']) & (v >= params['q75v']),
                (c >= params['q75c']) & (v <= params['q25v']))
    raise ValueError(rule)


def lexical_coefficient(c, lexical, params, runtime):
    """Reference for the proposed lexical controller; not installed in vLLM.

    Same confidence baseline and fitted amplitudes. Replace variance-dependent
    enhancement by the exact mixed-label rule. A lexical hit takes precedence
    over high confidence. This hard gate is an explicit experimental choice.
    """
    mid, k, a, b, _, _ = runtime._curve_constants(
        params.q25c, params.q75c, params.low_val_1, c.device, c.dtype, params.curve_tau)
    base = runtime._baseline(c, mid, k, a, b)
    over = lexical | (c < params.q25c)
    under = ~lexical & (c > params.q75c)
    result = torch.where(over, params.low_val_2, torch.where(under, params.high_val_2, base))
    return result.clamp(min=min(params.low_val_1, params.low_val_2),
                        max=max(.01, params.high_val_2))


def fit(features, over, under, original, output, runtime):
    if not over.any() or not under.any() or (over & under).any():
        raise ValueError('Empty or overlapping classes')
    xo, xu = features[over].astype(np.float64), features[under].astype(np.float64)
    mo, mu = xo.mean(0), xu.mean(0)
    direction = mo - mu
    w = direction / (xo.var(0) + xu.var(0) + 1e-12)
    projection = float(w @ direction)
    if not np.isfinite(projection) or projection <= 0:
        raise ValueError('Degenerate direction')
    midpoint = float(.5 * w @ (mo + mu))
    moderate = float((w @ mo - midpoint) / projection)
    aggressive = float(np.max((xo @ w - midpoint) / projection))
    assert np.isclose(moderate, .5) and aggressive >= moderate
    params = dict(original)
    params.update(low_val_1=-moderate, low_val_2=-aggressive,
                  curve_tau=min(.01, .5 * moderate * (1 - params['q75c']) /
                                (params['q75c'] - params['q25c'])))
    runtime.validate_curve_targets(params['q25c'], params['q75c'], params['low_val_1'], params['curve_tau'])
    controller = runtime.ReBalanceParams(boundary_token_ids=(1,), think_start_token_id=2,
                                        think_end_token_id=3, **params)
    c, v = torch.meshgrid(torch.linspace(0, 1, 501), torch.linspace(0, .25, 251), indexing='ij')
    coeff = runtime.compute_rebalance_coefficient(c, v, controller)
    assert torch.isfinite(coeff).all()
    assert coeff.min() >= min(params['low_val_1'], params['low_val_2']) - 1e-6
    assert coeff.max() <= max(.01, params['high_val_2']) + 1e-6
    vector = torch.from_numpy(direction.astype(np.float32))
    output.mkdir()
    torch.save(vector, output / 'auto_vector.pt')
    info = dict(parameters=params, positives=int(over.sum()), negatives=int(under.sum()),
                excluded=int((~(over | under)).sum()), hidden_state_index=21, decoder_output_layer=20,
                vector_norm=float(vector.double().norm()), vector_sha256=sha(output / 'auto_vector.pt'),
                curve_grid_points=coeff.numel(), curve_finite=True,
                method='Original raw mean difference and diagonal-LDA crossing fit; only class masks changed')
    save(output / 'fit.json', info)
    return vector, info, controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.main_root, args.output
    out.mkdir(parents=True, exist_ok=False)
    started_inputs = {}
    try:
        cache = root / '.codex_work/question_balanced_20260911/original_selected_layer'
        frozen = root / '.codex_work/auto_code_v2_500_20260908'
        tokenizer = root / '.codex_work/label_audit_30_20260910/tokenizer.json'
        archive = root / '.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
        author_path = root / 'sources/ReBalance/hidden_analysis_auto.py'
        runtime_path = root / 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
        paths = [cache/'steps.json', cache/'positions.json', cache/'layer_21.npy',
                 frozen/'fit.json', frozen/'auto_vector.pt', frozen/'protocol.json',
                 tokenizer, archive, author_path, runtime_path, Path(__file__), Path(__file__).with_name('policy.py')]
        started_inputs = {str(p): sha(p) for p in paths}
        save(out/'inputs.json', started_inputs)
        author = load('alignment_author', author_path)
        runtime = load('alignment_runtime', runtime_path)
        decoder = Decoder(tokenizer)
        for token, piece in TRIGGERS.items():
            assert decoder.decode([token]) == piece
        steps = json.loads((cache/'steps.json').read_text())
        positions = json.loads((cache/'positions.json').read_text())
        frozen_fit = json.loads((frozen/'fit.json').read_text())
        protocol = json.loads((frozen/'protocol.json').read_text())
        with tarfile.open(archive) as tar:
            raw = tar.extractfile('generations.jsonl').read()
            manifest = json.load(tar.extractfile('manifest.json'))
        assert hashlib.sha256(raw).hexdigest() == protocol['source_sha256']
        rows = [json.loads(line) for line in raw.splitlines()]
        del raw
        assert len(rows) == 500 and len(steps) == 84008 and manifest['temperature'] == 0
        assert [r['train_index'] for r in rows] == manifest['train_indices']
        big, small, train_ids, normalized_hashes = [], [], [], []
        import unicodedata
        for row in rows:
            text = row.get('problem', row.get('question'))
            assert isinstance(text, str), row.keys()
            normalized_hashes.append(hashlib.sha256(''.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest())
        expected_positions = [[] for _ in rows]
        max_confidence_error = 0.
        for step in steps:
            row = rows[step['question']]
            ids = row['token_ids'][step['start']:step['stop']]
            confidence = float(np.exp(np.asarray(row['logprobs'][step['start']:step['stop']], dtype=np.float64)).mean())
            # NumPy exp reductions can differ by FP64 rounding across hosts.
            # Keep the frozen values for all class thresholds; report the error.
            error = abs(confidence - step['confidence'])
            max_confidence_error = max(max_confidence_error, error)
            assert error <= 2e-15, (step['question'], step['start'], error)
            hit = bool(author.has_lexicon_hit(decoder.decode(ids)))
            assert hit == step['lexical_hit']
            big.append(hit)
            small.append(any(t in TRIGGERS for t in ids))
            train_ids.append(row['train_index'])
            expected_positions[step['question']].append(step['start'] + len(row['prompt_token_ids']))
        assert expected_positions == [r['positions'] for r in positions]
        c = np.asarray([r['confidence'] for r in steps])
        v = np.asarray([r['variance'] for r in steps])
        max_variance_error = 0.
        for i, step in enumerate(steps):
            expected = 0. if i == 0 or step['question'] != steps[i-1]['question'] else (float(c[i])-float(c[i-1]))**2/4
            error = abs(float(v[i])-expected)
            max_variance_error = max(max_variance_error, error)
            assert error <= 1e-16, (i, error)
        assert np.array_equal(np.quantile(c,[.25,.75]), protocol['confidence_quantiles'])
        assert np.array_equal(np.quantile(v,[.25,.75]), protocol['variance_quantiles'])
        big, small = np.asarray(big), np.asarray(small)
        features = np.load(cache/'layer_21.npy', mmap_mode='r')
        assert features.shape == (len(steps), 1536)
        baseline = torch.load(frozen/'auto_vector.pt', map_location='cpu', weights_only=True)
        results, masks = {}, {}
        for name, lex, rule in [('L27', big, 'lexical'), ('T14', small, 'lexical'),
                                ('CV', big, 'confidence_variance')]:
            over, under = labels(c, v, lex, frozen_fit['parameters'], rule)
            vector, info, control = fit(features, over, under, frozen_fit['parameters'], out/name, runtime)
            if name == 'L27':
                assert torch.equal(vector, baseline), 'Original vector not reproduced exactly'
                parameter_error = max(abs(info['parameters'][k]-v) for k,v in frozen_fit['parameters'].items())
                assert parameter_error < 1e-14, ('Original fit drift', parameter_error)
            info['cosine_to_frozen'] = float(torch.nn.functional.cosine_similarity(vector.double(), baseline.double(), dim=0).clamp(-1,1))
            info['norm_ratio_to_frozen'] = float(vector.double().norm()/baseline.double().norm())
            results[name] = info
            masks[name] = np.where(over, 1, np.where(under, -1, 0))
        with (out/'class_masks.npz').open('xb') as f:
            np.savez_compressed(f, **masks, big_lexical=big, small_lexical=small, train_indices=train_ids)
        p = runtime.ReBalanceParams(boundary_token_ids=(1,), think_start_token_id=2,
            think_end_token_id=3, **frozen_fit['parameters'])
        ct, vt, lt = torch.from_numpy(c), torch.from_numpy(v), torch.from_numpy(big)
        old = runtime.compute_rebalance_coefficient(ct, vt, p)
        new = lexical_coefficient(ct, lt, p, runtime)
        assert torch.isfinite(new).all()
        high_hit = lt & (ct > p.q75c)
        assert torch.all(new[high_hit] == p.low_val_2)
        report = dict(status='CPU calibration complete; no GPU efficacy test', question_count=500,
            steps=len(steps), original_vector_elementwise_equal=True,
            original_parameters_max_abs_error=parameter_error,
            control_uses_original_frozen_files=True,
            confidence_reconstruction_max_abs_error=max_confidence_error,
            variance_reconstruction_max_abs_error=max_variance_error,
            large_lexicon=author.LEXICON_BASE, small_token_vocabulary=TRIGGERS,
            lexical_overlap=dict(both=int((big&small).sum()), big_only=int((big&~small).sum()),
                                 small_only=int((~big&small).sum()), neither=int((~big&~small).sum())),
            variants=results,
            transitions={name:{f'{a}_to_{b}':int(((masks['L27']==a)&(mask==b)).sum())
                              for a in (-1,0,1) for b in (-1,0,1)} for name,mask in masks.items() if name!='L27'},
            lexical_controller_reference=dict(changed=int((old != new).sum()),
                positive_to_negative=int(((old>0)&(new<0)).sum()), negative_to_positive=int(((old<0)&(new>0)).sum()),
                explicit_design='Hard mixed-label gates; confidence baseline retained on middle nonlexical steps; not native-integrated'),
            limitations=['Calibration class counts and vector geometry are not compression or accuracy results.',
                'T14 uses exact token membership, L27 uses author regex with morphology; matcher differences are explicit.',
                'All variants use the same arithmetic confidence, saved step positions, raw vector convention and selected layer.',
                'Changing labels refits both raw vector and LDA crossing amplitudes; they are one calibration factor.',
                'CV changes only calibration labels; not a restoration of the abandoned paper reconstruction.'])
        save(out/'report.json', report)
        save(out/'data_usage.json', dict(purpose='Reuse original calibration only; no new development/confirmation split frozen',
            rows=[dict(train_index=r['train_index'],problem_sha256=h,purpose='calibration') for r,h in zip(rows,normalized_hashes)]))
        assert all(sha(p)==h for p,h in started_inputs.items()), 'Input changed during analysis'
        save(out/'complete.json', dict(inputs_unchanged=True, report_sha256=sha(out/'report.json')))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as exc:
        save(out/'failure.json', dict(error=repr(exc), inputs=started_inputs))
        raise


if __name__ == '__main__':
    main()
