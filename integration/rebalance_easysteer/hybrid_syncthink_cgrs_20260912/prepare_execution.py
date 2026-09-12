"""Freeze B data only after reconciling materialized A/B evidence; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import unicodedata
import zipfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def phash(s):
    return digest(''.join(unicodedata.normalize('NFKC', s).split()).encode())


def save(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    reg_path = HERE/'cpu_run1/data_registry.json'
    reg = json.loads(reg_path.read_text(encoding='utf-8'))
    reservations = reg['proposed_reservations']
    ids = {r['train_index'] for r in reservations}
    hashes = {r['problem_sha256'] for r in reservations}
    train_path = a.source_root/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(s) for s in train_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert len(ids) == len(hashes) == 272
    assert all(phash(train[r['train_index']]['problem']) == r['problem_sha256'] for r in reservations)
    scans, seen, hits, errors = {}, set(), [], []
    exclusions = {'sources', 'upstream', 'original_snapshot', 'node_modules', '.git'}

    def inspect(obj, label):
        if isinstance(obj, dict):
            match = (obj.get('train_index') in ids or
                     any(isinstance(obj.get(k), str) and phash(obj[k]) in hashes
                         for k in ('problem', 'question')))
            if match:
                hits.append(dict(path=label, train_index=obj.get('train_index'), keys=sorted(obj)))
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    inspect(value, label)
        elif isinstance(obj, list):
            for value in obj:
                if isinstance(value, (dict, list)):
                    inspect(value, label)

    def scan(label, data):
        h = digest(data)
        scans[label] = h
        if h in seen:
            return
        seen.add(h)
        if not data and 'migration_15536_20260911' in label and 'assets' in label:
            return
        try:
            text = data.decode('utf-8-sig')
            if label.endswith('.jsonl'):
                for line in text.splitlines():
                    if line.strip():
                        inspect(json.loads(line), label)
            else:
                inspect(json.loads(text), label)
        except (ValueError, UnicodeError) as exc:
            errors.append(dict(path=label, error=str(exc)))

    for directory in (a.source_root/'integration/rebalance_easysteer/configs', a.source_root/'.codex_work'):
        for path in directory.rglob('*'):
            if not path.is_file() or any(part in exclusions for part in path.relative_to(directory).parts):
                continue
            label = str(path.relative_to(a.source_root))
            if path.name == 'tokenizer.json':
                continue
            if path.suffix in ('.json', '.jsonl'):
                scan(label, path.read_bytes())
            elif path.name.endswith(('.tar.gz', '.tgz', '.zip')):
                if path.name.endswith('.zip'):
                    with zipfile.ZipFile(path) as z:
                        for name in z.namelist():
                            if name.endswith(('.json', '.jsonl')) and '/sources/' not in '/'+name:
                                scan(label+'::'+name, z.read(name))
                else:
                    with tarfile.open(path) as t:
                        for info in t:
                            if info.isfile() and info.name.endswith(('.json', '.jsonl')) and '/sources/' not in '/'+info.name:
                                scan(label+'::'+info.name, t.extractfile(info).read())
    # B's prior 100 read traces and original exclusions were checked in cpu_run1.
    assert not ids & {r['train_index'] for r in reg['used']}
    receipt = dict(registry_sha256=digest(reg_path.read_bytes()), scanned_files=len(scans),
                   unique_contents=len(seen), hits=hits, parse_errors=errors,
                   source_sha256=scans, global_local_snapshot_clear=not hits and not errors,
                   scope='A worktree configurations, downloaded outcomes and archive members; B registry used set; excludes full upstream data/code. Remote-only claims checked separately before execution.',
                   source_train_LF_sha256=digest(train_path.read_bytes().replace(b'\r\n', b'\n')))
    save(a.output/'data_audit.json', receipt)
    if hits or errors:
        raise RuntimeError(f'Data not cleared: {len(hits)} hits, {len(errors)} parse errors')
    for role in ('engineering', 'screening'):
        with (a.output/(role+'.jsonl')).open('x', encoding='utf-8', newline='\n') as f:
            for row in reservations:
                if row['purpose'] == role:
                    record = dict(train[row['train_index']], train_index=row['train_index'], problem_sha256=row['problem_sha256'])
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
    save(a.output/'data_freeze.json', dict(status='local_data_frozen_pending_remote_inventory',
         registry_sha256=receipt['registry_sha256'], audit_sha256=digest((a.output/'data_audit.json').read_bytes()),
         role_files={role: digest((a.output/(role+'.jsonl')).read_bytes()) for role in ('engineering','screening')},
         confirmation='200 reserved; no prompts or answers sent to model', reservations=reservations))
    print(json.dumps({k:receipt[k] for k in ('scanned_files','unique_contents','global_local_snapshot_clear')}))


if __name__ == '__main__':
    main()
