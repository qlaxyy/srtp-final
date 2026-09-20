"""Read-only CPU audit of saved production and eager R trajectories.

No model loads, fitting, generation, or changes to frozen artifacts.
First divergence is zero based. A boundary at b can affect output b+1.
"""
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    home = Path(__file__).resolve().parent
    root = next(p for p in home.parents if (p / '.git').exists())
    work = root / '.codex_work'
    paths = {
        'production': work / 'iterative_recalibration_20260919/calibration_evidence.tar.gz',
        'observed': work / 'online_endpoint_full_20260919/evidence.tar.gz',
        'tokenizer': work / 'wsc_weights_20260917/tokenizer.json',
    }
    tok = json.loads(paths['tokenizer'].read_text(encoding='utf-8'))
    bounds = {i for s, i in tok['model']['vocab'].items() if 'ĊĊ' in s}
    with tarfile.open(paths['production']) as archive:
        old = json.load(archive.extractfile('R_train500/result.json'))['records']
    batches = {}
    with tarfile.open(paths['observed'], 'r|gz') as archive:
        for member in archive:
            parts = member.name.split('/')
            if parts[0] == 'online_endpoint_full_20260919_collect_run1' and len(parts) >= 2 and parts[-1] == 'result.json' and parts[-2].startswith('R_batch'):
                offset = int(parts[-2][len('R_batch'):])
                assert offset not in batches
                batches[offset] = json.load(archive.extractfile(member))['records']
    new = [r for offset in sorted(batches) for r in batches[offset]]
    assert len(old) == len(new) == 500
    counts = dict(exact=0, before_any_possible_R_effect=0,
                  at_first_possible_R_effect=0, after_first_possible_R_effect=0,
                  prefix_only=0)
    details = []
    for q, (a, b) in enumerate(zip(old, new)):
        assert a['problem_sha256'] == b['problem_sha256']
        x, y = a['token_ids'], b['token_ids']
        k = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        boundary = next((i for i, t in enumerate(x[:k]) if t in bounds), None)
        if x == y:
            category = 'exact'
        elif k == min(len(x), len(y)):
            category = 'prefix_only'
        elif boundary is None:
            category = 'before_any_possible_R_effect'
        elif k == boundary + 1:
            category = 'at_first_possible_R_effect'
        else:
            category = 'after_first_possible_R_effect'
        counts[category] += 1
        details.append(dict(question=q, category=category, shared_prefix_tokens=k,
                            first_shared_boundary=boundary, old_tokens=len(x), new_tokens=len(y)))
    result = dict(scope='Saved R-only full calibration500, production versus eager observer; CPU only',
                  counts=counts, details=details,
                  interpretation='Divergence before any shared generated boundary cannot be caused by direct ReBalance injection in that request. Later divergence is not proof of a steering bug. Multiple execution settings changed together.',
                  inputs={key:dict(path=str(p), sha256=sha(p)) for key,p in paths.items()})
    output = home / 'execution_audit_20260920.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
