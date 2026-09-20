"""Fit q75/q90 on identical saved native states, CPU only; no generation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import numpy as np

HOME = Path(__file__).resolve().parent
ROOT = next(p for p in HOME.parents if (p / '.git').exists())
HERE = HOME.parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    archive = ROOT / '.codex_work/online_endpoint_full_20260919/evidence.tar.gz'
    archive_hash = sha(archive)
    assert archive_hash == '210462946624e2447ab13e7db9e9f9785b67edd4812a62403f04c02cbeeb2233'
    sources = {}
    # Stream once. Do not extract arbitrary archive paths onto disk.
    with tarfile.open(archive, 'r|gz') as t:
        for member in t:
            parts = member.name.split('/')
            if not parts[0] in ['online_endpoint_full_20260919_collect_run1',
                                'online_endpoint_full_20260919_recovery52_run1']:
                continue
            name = parts[-1]
            if name in ['R_steps.json', 'L27_steps.json', 'R_labels.json', 'L27_labels.json']:
                source = name.split('_')[0]
                sources.setdefault(source, {})[name.split('_')[1].split('.')[0]] = json.load(t.extractfile(member))
            elif len(parts) == 3 and name.endswith('_hidden.npz'):
                source = parts[1].split('_')[0]
                if source not in ['R', 'L27']:
                    continue
                q = int(name.split('_')[0])
                group = sources.setdefault(source, {}).setdefault('states', {})
                assert q not in group
                with np.load(io.BytesIO(t.extractfile(member).read())) as z:
                    assert np.array_equal(z['pre_hidden'], z['post_hidden'])
                    group[q] = (z['starts'].copy(), z['pre_hidden'].copy())
    sys.path.insert(0, str(ROOT / 'integration/rebalance_easysteer/scripts'))
    import calibrate_auto as cal
    import torch
    torch.set_num_threads(8)
    report = {'archive_sha256': archive_hash, 'new_answers': 0, 'model_forwards': 0,
              'measurement': 'saved native eager/split pre states, hidden21/decoder20 fixed',
              'groups': {}, 'calibration_code_sha256': sha(Path(cal.__file__))}
    for source in ['R', 'L27']:
        data = sources[source]
        assert len(data['states']) == len(data['labels']) == 500
        steps = data['steps']
        indices = {q: {int(start): i for i, start in enumerate(z[0])} for q,z in data['states'].items()}
        x = np.stack([data['states'][s['question']][1][indices[s['question']][s['start']]] for s in steps])
        assert np.isfinite(x).all()
        folder = a.output / source
        folder.mkdir()
        np.save(folder / 'layer_21.npy', x)
        c = np.array([s['confidence'] for s in steps])
        v = np.array([s['variance'] for s in steps])
        lex = np.array([s['lexical_hit'] for s in steps])
        q = np.array([s['question'] for s in steps])
        q25 = float(np.quantile(c, .25))
        over = lex | (c < q25)
        results, directions = {}, {}
        for quantile in [.75, .90]:
            name = 'q' + str(round(quantile * 100))
            out = folder / name
            out.mkdir()
            threshold = float(np.quantile(c, quantile))
            under = (~lex) & (c > threshold)
            assert under.any() and not np.any(over & under)
            save(out / 'steps.json', steps)
            save(out / 'protocol.json', dict(source_sha256=archive_hash,
                 confidence_quantiles=[q25, threshold], variance_quantiles=np.quantile(v,[.25,.75]).tolist(),
                 confidence_quantile_levels=[.25, quantile], high_threshold_used_for_both_label_and_curve=True))
            save(out / 'selected_layer.json', {'best': {'layer':21,'selection':'fixed existing native capture layer; no automatic selection'}})
            save(out / 'collection.json', {'feature_dir':str(folder.resolve()), 'layer_ids':[21]})
            cal.fit(SimpleNamespace(output=out, model=Path('/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B')))
            fit = read(out / 'fit.json')
            assert fit['parameters']['q75c'] == threshold
            assert fit['negatives'] == int(under.sum()) and fit['positives'] == int(over.sum())
            # Correct legacy auto-fit metadata: this experiment fixes the observed layer.
            fit.update(version='native-under-quantile-20260920', confidence_high_quantile=quantile,
                method='Saved intervened native states; fixed hidden21; lexical OR low-confidence O, nonlexical AND high-confidence U; raw mean O-U; original dynamic fitting formula with common high-confidence quantile.',
                legacy_parameter_note='q75c is a runtime field name; q90 candidate stores the empirical 90th percentile here. q75v remains variance 75th percentile.')
            (out / 'fit.json').write_bytes((json.dumps(fit,ensure_ascii=False,indent=2)+'\n').encode())
            counts = np.bincount(q[under],minlength=500)
            results[name] = dict(threshold=threshold, under_steps=int(under.sum()),
                under_questions=int((counts>0).sum()), over_steps=int(over.sum()),
                under_from_capped=sum(int(data['labels'][i]['capped']) for i in q[under]),
                under_from_wrong=sum(int(not data['labels'][i]['correct']) for i in q[under]),
                top10_steps=int(np.sort(counts)[-10:].sum()), vector_norm=fit['vector_norm'],
                parameters=fit['parameters'], curve_check=read(out/'curve_check.json'),
                vector_sha256=sha(out/'auto_vector.pt'), fit_sha256=sha(out/'fit.json'))
            directions[name] = x[over].mean(0,dtype=np.float64)-x[under].mean(0,dtype=np.float64)
        d0,d1=directions['q75'],directions['q90']
        results['cosine'] = float(d0@d1/(np.linalg.norm(d0)*np.linalg.norm(d1)))
        assert results['q90']['under_steps'] <= results['q75']['under_steps']
        report['groups'][source] = results
    save(a.output/'summary.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
