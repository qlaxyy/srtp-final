"""CPU-only frozen calibration label audit; no refit or model execution."""
import argparse
import hashlib
import json
import tarfile
import unicodedata
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    base = a.root / '.codex_work'
    paths = dict(steps=base/'question_balanced_20260911/original_selected_layer/steps.json',
                 protocol=base/'auto_code_v2_500_20260908/protocol.json',
                 fit=base/'auto_code_v2_500_20260908/fit.json',
                 archive=base/'own_calibration_500_20260908/calibration_assets.tar.gz',
                 tokenizer=base/'label_audit_30_20260910/tokenizer.json', script=Path(__file__))
    protocol = json.loads(paths['protocol'].read_text())
    fit = json.loads(paths['fit'].read_text())
    steps = json.loads(paths['steps'].read_text())
    tok = json.loads(paths['tokenizer'].read_text(encoding='utf8'))
    assert any(t['id'] == 151649 and t['content'] == '</think>' for t in tok['added_tokens'])
    with tarfile.open(paths['archive']) as tf:
        raw = tf.extractfile('generations.jsonl').read()
        manifest = json.load(tf.extractfile('manifest.json'))
    assert hashlib.sha256(raw).hexdigest() == protocol['source_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 500 and len(steps) == 84008
    assert [r['train_index'] for r in rows] == manifest['train_indices']
    length = np.array([r['token_ids'].index(151649) if 151649 in r['token_ids'] else len(r['token_ids']) for r in rows])
    q = np.array([s['question'] for s in steps])
    c = np.array([s['confidence'] for s in steps])
    hit = np.array([s['lexical_hit'] for s in steps], dtype=bool)
    assert all(s['stop'] <= length[s['question']] for s in steps)
    assert np.array_equal(np.quantile(c, [.25, .75]), protocol['confidence_quantiles'])
    low, high = protocol['confidence_quantiles']
    over, under = hit | (c < low), ~hit & (c > high)
    assert int(over.sum()) == fit['positives'] and int(under.sum()) == fit['negatives']

    def counts(mask):
        n = np.bincount(q[mask], minlength=500)
        return dict(steps=int(mask.sum()), questions=int((n > 0).sum()),
                    largest_question_share=float(n.max()/n.sum()) if n.sum() else None,
                    top10_question_share=float(np.sort(n)[-10:].sum()/n.sum()) if n.sum() else None)

    def compare(mask, gate):
        kept = mask & gate[q]
        return dict(original=counts(mask), retained=counts(kept), removed=counts(mask & ~gate[q]),
                    removed_percent=float(100*(mask & ~gate[q]).sum()/mask.sum()))

    variants = {}
    for name, lens in [('thinking', length), ('total_completion_sensitivity', np.array([len(r['token_ids']) for r in rows]))]:
        mean = float(lens.mean())
        variants[name] = dict(mean=mean, median=float(np.median(lens)), minimum=int(lens.min()), maximum=int(lens.max()),
            above=int((lens > mean).sum()), below=int((lens < mean).sum()), equal=int((lens == mean).sum()),
            over=compare(over, lens > mean), under=compare(under, lens < mean),
            strict_low_AND_lexical_over=compare(hit & (c < low), lens > mean))
    registry = []
    for i, r in enumerate(rows):
        text = r.get('problem', r.get('question'))
        h = hashlib.sha256(''.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()
        registry.append(dict(train_index=r['train_index'], problem_sha256=h, thinking_tokens=int(length[i]),
            total_tokens=len(r['token_ids']), has_think_end=151649 in r['token_ids'],
            finish_reason=r.get('finish_reason'), purpose='reuse original calibration only; no evaluation selection',
            over_original=int((over & (q == i)).sum()), under_original=int((under & (q == i)).sum()),
        ))
    result = dict(model=protocol['model'], source_sha256=protocol['source_sha256'],
        input_sha256={k: digest(v) for k,v in paths.items()},
        scope='Frozen 1.5B original 500 calibration answers; strict length gates; no relabel thresholds, refit, layer selection, inference, or correctness claim',
        confidence_thresholds=dict(q25=low,q75=high),
        original_rule='over = lexical OR confidence<q25; under = NOT lexical AND confidence>q75',
        no_think_end=sum(not r['has_think_end'] for r in registry),
        generation_manifest=manifest, variants=variants, questions=registry)
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/'result.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('questions','generation_manifest','input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
