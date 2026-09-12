"""Read-only inventory of local outcome exposure and both lines' data claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import tarfile
import zipfile
from prepare_bcc import read, save, sha, require


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--peer-registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    require(not a.output.exists(), 'Output exists')
    cfg = root/'integration/rebalance_easysteer/configs'
    target = cfg/'mechanism_screen100_20260911/confirmation200.jsonl'
    rows = [json.loads(x) for x in target.read_text(encoding='utf-8').splitlines()]
    plan = read(target.parent/'plan.json'); require(sha(target) == plan['confirmation']['dataset_sha256'], 'Reserve hash')
    ids = {r['train_index'] for r in rows}
    texts = {normalized(r['problem']) for r in rows}
    hashes = {hashlib.sha256(t.encode()).hexdigest() for t in texts}
    require(len(ids) == len(texts) == 200, 'Nonunique reserve')
    train_path = root/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(x) for x in train_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    for r in rows: require(normalized(r['problem']) == normalized(train[r['train_index']]['problem']), 'Train ID identity')
    excluded = set().union(*(set(v) for v in plan['exclusions'].values()))
    checks = {'original_exclusion_overlap': sorted(ids & excluded)}
    for dataset in ['Math_Math500', 'Math_GSM8K']:
        pp = root/'sources/ReBalance/Data'/dataset/'test.jsonl'
        rr = [json.loads(x) for x in pp.read_text(encoding='utf-8').splitlines() if x.strip()]
        checks[dataset+'_normalized_overlap'] = sum(normalized(r['problem']) in texts for r in rr)
    peer = read(a.peer_registry); peer_sha = sha(a.peer_registry)
    for kind in ['used', 'proposed_reservations']:
        checks['peer_'+kind+'_overlap'] = [r['train_index'] for r in peer[kind]
            if r['train_index'] in ids or r['problem_sha256'] in hashes]
    bcc = read(cfg/'bcc_v1_20260912/candidate_manifest.json')
    checks['bcc_overlap'] = sorted(ids & {r['train_index'] for g in bcc['groups'].values() for r in g['details']})
    # Ignore third-party source/data and damaged historical extraction copies;
    # inspect all materialized local experimental JSON, including partial runs.
    excluded_dirs = {'sources', 'upstream', 'original_snapshot', 'node_modules', '.git'}
    paths = sorted(set(cfg.rglob('*.json')) | set(cfg.rglob('*.jsonl')) |
                   {p for p in (root/'.codex_work').rglob('*') if p.suffix in ('.json','.jsonl')
                    and not any(v in excluded_dirs for v in p.relative_to(root/'.codex_work').parts)})
    sources = {}; seen = {}; exposure = []; parse_errors = []; reservation_hits = []; empty_metadata = []
    def inspect(obj, path):
        if isinstance(obj, dict):
            match = obj.get('train_index') in ids or any(isinstance(obj.get(k), str) and normalized(obj[k]) in texts for k in ['problem','question'])
            if match and (isinstance(obj.get('correct'), bool) or any(k in obj for k in ['token_ids','generated_text','prediction'])):
                exposure.append(dict(path=path, train_index=obj.get('train_index'), keys=sorted(obj.keys())))
            if match and 'train_index' in obj and 'problem' in obj and 'solution' in obj:
                reservation_hits.append(dict(path=path, train_index=obj['train_index']))
            for v in obj.values():
                if isinstance(v, (dict,list)): inspect(v,path)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)): inspect(v,path)
    for pp in paths:
        if pp.name == 'tokenizer.json': continue
        data = pp.read_bytes(); digest = hashlib.sha256(data).hexdigest(); rel = str(pp.relative_to(root))
        sources[rel] = digest
        if not data and pp.parent.name == 'migration_15536_20260911' and pp.name in ('assets.json','assets_attempt2.json'):
            empty_metadata.append(dict(path=rel,sha256=digest,reason='Zero-byte historical migration asset manifest; no answer content. Kept, not repaired.'))
            continue
        if digest in seen: continue
        seen[digest] = rel
        try:
            text = data.decode('utf-8-sig')
            if pp.suffix == '.jsonl':
                for line in text.splitlines():
                    if line.strip(): inspect(json.loads(line),rel)
            else: inspect(json.loads(text),rel)
        except (ValueError,UnicodeError) as exc: parse_errors.append(dict(path=rel,error=str(exc)))
    archives=[]
    for pp in (root/'.codex_work').rglob('*'):
        if not pp.name.endswith(('.tar.gz','.tgz','.zip')) or any(v in excluded_dirs for v in pp.relative_to(root/'.codex_work').parts):continue
        rel=str(pp.relative_to(root)); archives.append(dict(path=rel,sha256=sha(pp)))
        def member(name,data):
            digest=hashlib.sha256(data).hexdigest();label=rel+'::'+name;sources[label]=digest
            if digest in seen:return
            seen[digest]=label
            try:
                text=data.decode('utf-8-sig')
                if name.endswith('.jsonl'):
                    for line in text.splitlines():
                        if line.strip():inspect(json.loads(line),label)
                else:inspect(json.loads(text),label)
            except (ValueError,UnicodeError) as exc:parse_errors.append(dict(path=label,error=str(exc)))
        if pp.name.endswith('.zip'):
            with zipfile.ZipFile(pp) as z:
                for info in z.infolist():
                    if not info.is_dir() and info.filename.endswith(('.json','.jsonl')):member(info.filename,z.read(info))
        else:
            with tarfile.open(pp) as t:
                for info in t:
                    if info.isfile() and info.name.endswith(('.json','.jsonl')):
                        with t.extractfile(info) as f:member(info.name,f.read())
    require(sha(a.peer_registry) == peer_sha, 'Peer registry changed during audit')
    save(a.output, dict(status='local_snapshot_audited_not_server_authorization', target_sha256=sha(target),
        peer_registry_sha256=peer_sha, peer_registry=str(a.peer_registry), train_sha256=sha(train_path),
        checks=checks, outcome_exposure=exposure, parse_errors=parse_errors,
        known_local_identity_and_exposure_clear=not any(checks.values()) and not exposure and not parse_errors,
        scanned_files=len(sources), unique_contents=len(seen), source_sha256=sources,
        archives=archives,empty_metadata=empty_metadata,
        dataset_definition_hits=reservation_hits,
        scope='Materialized experimental JSON/JSONL and all local experimental ZIP/tar.gz members in current configs and .codex_work, including partial outputs and historical worktree configs; third-party source/upstream and damaged duplicate extraction excluded. Remote-only artifacts not inspected; does not assert that unreported remote jobs never ran.',
        new_answers=0, GPU_calls=0, script_sha256=sha(Path(__file__))))
    print(dict(checks=checks, exposures=len(exposure), errors=len(parse_errors), files=len(sources)))


if __name__ == '__main__': main()
