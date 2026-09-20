"""CPU-only benchmark inventory and cross-worktree exposure audit, not a GPU runner."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def phash(text):
    return digest(''.join(unicodedata.normalize('NFKC', text).split()).encode())


def boxed(solution):
    """Keep the raw, balanced sole reference box; no generated-answer judging."""
    assert solution.count('\\boxed') == 1
    start = solution.index('\\boxed') + len('\\boxed')
    while solution[start].isspace():
        start += 1
    assert solution[start] == '{'
    depth = 1
    for end in range(start + 1, len(solution)):
        if solution[end] == '{':
            depth += 1
        elif solution[end] == '}':
            depth -= 1
        if depth == 0:
            answer = solution[start + 1:end]
            assert answer.strip()
            return answer
    raise ValueError('Unbalanced reference answer')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--source-dir', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    assert not args.output.exists(), 'Never overwrite an earlier proposal'
    sources = json.loads((args.source_dir/'receipt.json').read_text(encoding='utf8'))
    rows = []
    shape = {}
    for source in sources:
        name = source['name']
        raw = (args.source_dir/Path(source['path']).name).read_bytes()
        assert digest(raw) == source['sha256']
        data = json.loads(raw) if name == 'svamp' else [json.loads(x) for x in raw.splitlines() if x.strip()]
        assert len(data) == {'svamp': 1000, 'minerva_math': 272, 'olympiadbench': 675}[name]
        for i, item in enumerate(data):
            if name == 'svamp':
                problem = item['Body'].strip() + ' ' + item['Question'].strip()
                identity, gold = item['ID'], str(item['Answer'])
                meta = dict(type=item['Type'], answer_type='numeric')
            elif name == 'minerva_math':
                problem, identity, gold = item['problem'], item['idx'], boxed(item['solution'])
                meta = dict(type=item['type'], answer_type='reference_box_unmodified')
            else:
                assert not item['context'] and not re.search(r'<(?:img|image)', item['question'])
                assert len(item['final_answer']) == 1
                problem, identity, gold = item['question'], item['id'], item['final_answer'][0]
                meta = {key: item[key] for key in ('subfield', 'is_multiple_answer', 'unit', 'answer_type', 'error')}
            rows.append(dict(dataset=name, source_id=identity, dataset_index=i, problem=problem,
                             problem_sha256=phash(problem), reference_answer=gold, metadata=meta))
        shape[name] = dict(n=len(data), unique_ids=len({r['source_id'] for r in rows if r['dataset']==name}))
        assert shape[name]['unique_ids'] == len(data)
    targets = collections.defaultdict(list)
    for row in rows:
        targets[row['problem_sha256']].append((row['dataset'], row['source_id']))
    hits = collections.defaultdict(set)
    files, errors = {}, []
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(x) for x in train_path.read_text(encoding='utf8').splitlines() if x.strip()]
    files[str(train_path)] = digest(train_path.read_bytes())
    def inspect(obj, label):
        if isinstance(obj, dict):
            values = {phash(obj[k]) for k in ('problem', 'question') if isinstance(obj.get(k), str)}
            values.update(obj[k] for k in ('problem_sha256', 'normalized_prompt_sha256') if isinstance(obj.get(k), str))
            for k in ('train_index', 'train_idx'):
                i = obj.get(k)
                if type(i) is int and 0 <= i < len(train):
                    values.add(phash(train[i]['problem']))
            for h in values & targets.keys():
                hits[h].add(label)
            for value in obj.values():
                if isinstance(value, (list, dict)):
                    inspect(value, label)
        elif isinstance(obj, list):
            for value in obj:
                if isinstance(value, (list, dict)):
                    inspect(value, label)
    trees = [Path(x[9:]) for x in subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT, text=True).splitlines() if x.startswith('worktree ')]
    for tree in trees:
        for folder in (tree/'integration/rebalance_easysteer/configs', tree/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
            for path in folder.rglob('*'):
                if path.suffix not in ('.json', '.jsonl') or not path.is_file() or path.name == 'gsm8k_source_train.jsonl':
                    continue
                raw = path.read_bytes()
                files[str(path)] = digest(raw)
                try:
                    if path.suffix == '.json':
                        inspect(json.loads(raw.decode('utf-8-sig')), str(path))
                    else:
                        for line in raw.decode('utf-8-sig').splitlines():
                            if line.strip():
                                inspect(json.loads(line), str(path))
                except (ValueError, UnicodeError) as exc:
                    errors.append(dict(path=str(path), error=repr(exc)))
    for row in rows:
        row['local_prior_exposure'] = row['problem_sha256'] in hits
        row['purpose'] = 'proposed_fixed_policy_generalization_test_not_parameter_selection'
    for name in shape:
        shape[name]['locally_exposed_rows'] = sum(r['local_prior_exposure'] for r in rows if r['dataset']==name)
    assert not errors, errors
    args.output.mkdir(parents=True)
    def save(name, value):
        with (args.output/name).open('x', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    save('rows_proposed.json', rows)
    save('source_receipt.json', sources)
    save('cpu_audit.json', dict(datasets=shape, source_sha256=files, errors=errors,
        duplicate_question_groups=[dict(problem_sha256=h, ids=ids) for h, ids in targets.items() if len(ids)>1],
        prior_exposure=[dict(problem_sha256=h, proposed_ids=targets[h], evidence=sorted(paths)) for h, paths in hits.items()],
        normalization='NFKC, remove all Unicode whitespace, SHA256 UTF8',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        status='local_inventory_complete_not_frozen_not_gpu_ready',
        limitations=['No remote registry scan; no claim of semantic independence or absence of pretraining contamination',
                    'Conservative train_index handling may report false exposure matches for other dataset indices',
                    'Raw reference extraction only; generated-answer grader and numeric tolerances still require CPU acceptance',
                    'OlympiadBench is Qwen evaluation distribution, not all modalities/languages of the original benchmark']))
    print(json.dumps(dict(datasets=shape,duplicates=sum(len(x)>1 for x in targets.values()),scanned=len(files)),indent=2))


if __name__ == '__main__':
    main()
