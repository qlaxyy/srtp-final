"""Isolated iterative recalibration; never overwrite a run or frozen asset."""
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parent
HERE = HOME.parent
ROOT = HERE.parents[3]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def spans(ids, boundaries, end=151649):
    closed = end in ids
    n = ids.index(end) if closed else len(ids)
    result, first = [], 0
    for stop in range(n + 1):
        if not (stop < n and ids[stop] in boundaries) and not (stop == n and closed):
            continue
        if stop > first:
            result.append((first, stop))
        first = stop + 1
    return result, n

def verify():
    m = read(HOME / 'manifest.json')
    for name, digest in m['files'].items():
        if sha(HOME / name) != digest:
            raise ValueError('Changed input: ' + name)
    for name, digest in m['repo_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Changed dependency: ' + name)
    return m
