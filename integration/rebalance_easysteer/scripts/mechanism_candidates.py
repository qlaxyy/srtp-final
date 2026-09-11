"""CPU-only geometry and fixed checkpoint-cost utilities; no torch/model import."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = 'integration/rebalance_easysteer/'
ORIGINAL_PT_SHA = 'fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, source=False):
    if source:
        return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def read_vector(path, template):
    """Read only the verified tensor's storage, never execute pickle instructions."""
    require(sha(template) == ORIGINAL_PT_SHA, 'Unrecognized tensor template')
    with zipfile.ZipFile(template) as original, zipfile.ZipFile(path) as archive:
        names = original.namelist()
        require(names == ['auto_vector/data.pkl', 'auto_vector/byteorder', 'auto_vector/data/0',
                          'auto_vector/version', 'auto_vector/.data/serialization_id'], 'Unknown layout')
        require(archive.namelist() == names and archive.testzip() is None, 'Damaged tensor archive')
        for name in names:
            if name != 'auto_vector/data/0':
                require(archive.read(name) == original.read(name), 'Tensor schema changed: '+name)
        raw = archive.read('auto_vector/data/0')
        require(len(raw) == 1536*4, 'Wrong storage size')
        vector = np.frombuffer(raw, dtype='<f4').copy()
        require(np.isfinite(vector).all(), 'Nonfinite vector')
        return vector


def write_vector(path, vector, template):
    """Clone the known torch.save schema and replace its single FP32 storage.

    zip CRCs are regenerated. Shape/dtype/stride pickle bytes stay unchanged.
    A real torch.load + EasySteer payload check is additionally required in the
    existing server runtime before loading a model; CPU archive checks do not
    claim to be that acceptance test.
    """
    read_vector(template, template)
    vector = np.asarray(vector, dtype='<f4')
    require(vector.shape == (1536,) and np.isfinite(vector).all(), 'Invalid direction')
    with zipfile.ZipFile(template) as original, zipfile.ZipFile(path, 'x') as archive:
        for info in original.infolist():
            raw = vector.tobytes() if info.filename == 'auto_vector/data/0' else original.read(info.filename)
            archive.writestr(info, raw)
    require(np.array_equal(read_vector(path, template), vector), 'Storage round trip failed')


def bfloat16(value):
    """Round finite FP32 values to BF16, ties-to-even; return FP32 containers."""
    value = np.asarray(value, dtype=np.float32)
    require(np.isfinite(value).all(), 'BF16 emulation needs finite input')
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def cosine(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def moments(x, masks, chunk=2048):
    """Bounded-memory population moments; mask rows, never an entire FP64 copy."""
    accum = {name: [0, np.zeros(x.shape[1]), np.zeros(x.shape[1])] for name in masks}
    for start in range(0, len(x), chunk):
        part = np.asarray(x[start:start+chunk], dtype=np.float64)
        for name, mask in masks.items():
            selected = part[mask[start:start+chunk]]
            if not len(selected):
                continue
            count, mean, m2 = accum[name]
            local_mean = selected.mean(axis=0)
            delta = local_mean-mean
            new_count = count+len(selected)
            local_m2 = ((selected-local_mean)**2).sum(axis=0)
            accum[name] = [new_count, mean+delta*(len(selected)/new_count),
                           m2+local_m2+delta**2*(count*len(selected)/new_count)]
    return {name: dict(count=n, mean=mean, variance=m2/n) for name, (n, mean, m2) in accum.items() if n}


def directions(over, under):
    d = over['mean']-under['mean']
    w = d/(over['variance']+under['variance']+1e-12)
    score = float(w@d)
    require(score > 0 and np.isfinite(w).all(), 'Degenerate readout')
    mean = (over['count']*over['mean']+under['count']*under['mean'])/(over['count']+under['count'])
    require(np.linalg.norm(mean) > 0, 'Zero background mean')
    u = mean/np.linalg.norm(mean)
    residual = d-u*(u@d)
    denominator = float(w@residual)
    minimum = w*(score/(w@w))
    orthogonal = residual*(score/denominator) if denominator > 0 else None
    return dict(original=d, min_displacement=minimum, orthogonal_mean=orthogonal,
                w=w, u=u, score=score, denominator_ratio=denominator/score)


def checkpoint_cost(length, start=1024, stride=1024, probe_cap=48):
    """A fixed sparse policy's upper probe-token accounting, not speed prediction."""
    checkpoints = list(range(start, int(length), stride))
    return dict(checkpoints=len(checkpoints), max_extra_probe_tokens=len(checkpoints)*probe_cap,
                repeated_prefill_prefix_tokens=sum(checkpoints),
                earliest_probe_break_even_remaining_tokens=probe_cap if checkpoints else None)


def zero_loss_upper(n, delta=0.05):
    require(n > 0 and 0 < delta < 1, 'Invalid confidence-bound arguments')
    return float(-np.expm1(np.log(delta)/n))
