"""Bind executor-supplied reconciled training rows to a unique run plan."""
import argparse
import hashlib
import json
from pathlib import Path

from policy import problem_hash
from run import HERE, read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--coordination', type=Path, required=True)
    parser.add_argument('--phase', choices=['engineering', 'screen'], required=True)
    parser.add_argument('--dataset', choices=['math', 'gsm8k'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    rows = [json.loads(s) for s in a.rows.read_text(encoding='utf8').splitlines()]
    n = 8 if a.phase == 'engineering' else 64
    if len(rows) != n:
        raise ValueError(f'Expected exactly {n} rows')
    for row in rows:
        if row['split'] != 'train' or row['problem_sha256'] != problem_hash(row['problem']):
            raise ValueError('Invalid training row identity')
    if len({r['problem_sha256'] for r in rows}) != n:
        raise ValueError('Duplicate normalized problem')
    coord = read(a.coordination)
    if (coord['rows_sha256'] != sha(a.rows) or coord['run_id'] != a.run_id
            or coord['dataset'] != a.dataset or coord['phase'] != a.phase
            or not coord['data_reconciled'] or not coord['gpu_authorized']):
        raise ValueError('Executor coordination receipt does not bind this batch')
    if not a.run_id.startswith('cgrs_probe_') or Path(a.run_id).name != a.run_id:
        raise ValueError('Invalid independent run_id')
    identity = read(HERE/'identity.json')
    gate = None if a.gate is None else dict(path=str(a.gate), sha256=sha(a.gate))
    if a.phase == 'screen' and gate is None:
        raise ValueError('Screening needs completed engineering evidence')
    save(a.output, dict(phase=a.phase, dataset=a.dataset, run_id=a.run_id,
                        rows=rows, coordination=coord, engineering_gate=gate,
                        rows_sha256=sha(a.rows), assets=identity['assets'],
                        source_sha256=identity['source_sha256'],
                        implementation_commit=identity['base_commit']))


if __name__ == '__main__':
    main()
