"""Audit consumed training traces and propose new questions; never freeze data."""
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata
from build_patch import HERE, ROOT


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def phash(s):
    s = ''.join(unicodedata.normalize('NFKC', s).split())
    return hashlib.sha256(s.encode()).hexdigest()


def save(name, obj):
    if (HERE/name).exists():
        assert read(HERE/name) == obj, 'Refuse changed rerun: ' + name
        return
    with (HERE/name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    ns = HERE.parent
    archive = ROOT/'.codex_work/hybrid_expanded1_verified/results/easysteer'
    source = archive/'hybrid_syncthink_cgrs_20260912/s64_confirm_math200_gsm200_async_run1_20260913'
    audit = {}; usage = []; hashes = {}
    for role in ('math', 'gsm8k'):
        values = {}; labels = {}
        previous = read(ns/f'expanded_20260913/results/{role}_analysis.json')
        for arm in ('R', 'RS'):
            p = source/role/arm/'result.json'; g = p.with_name('author_grade.json')
            assert sha(p) == previous['source_sha256'][arm]['result']
            assert sha(g) == previous['source_sha256'][arm]['grade']
            raw, grade = read(p), read(g)
            assert grade['input_sha256'] == sha(p)
            values[arm], labels[arm] = raw['records'], grade['records']
            assert len(values[arm]) == len(labels[arm]) == 200
            hashes[str(p.relative_to(ROOT))] = sha(p)
            hashes[str(g.relative_to(ROOT))] = sha(g)
        rows = []
        for a, b, x, y in zip(values['R'], values['RS'], labels['R'], labels['RS']):
            assert a['train_index'] == b['train_index'] == x['train_index'] == y['train_index']
            assert phash(a['problem']) == phash(b['problem'])
            h = b['hybrid']; harmed = x['author_correct'] and not y['author_correct']
            rows.append(dict(train_index=a['train_index'], problem_sha256=phash(a['problem']),
                             R_correct=x['author_correct'], RS_correct=y['author_correct'],
                             harmed=bool(harmed), first_rank=h['first_rank'],
                             first_entropy=h['first_entropy'], first_trigger=h['first_trigger']))
            usage.append(dict(dataset=role+'_train', train_index=a['train_index'],
                              problem_sha256=phash(a['problem']),
                              purpose='post_failure_mechanism_audit_soft2_development',
                              status='already_consumed_no_new_generation'))
        audit[role] = dict(n=200, triggered=sum(r['first_trigger'] >= 0 for r in rows),
                          harmed=sum(r['harmed'] for r in rows),
                          harmed_non_top1=sum(r['harmed'] and r['first_rank'] > 0 for r in rows),
                          all_trigger_non_top1=sum(r['first_trigger'] >= 0 and r['first_rank'] > 0 for r in rows),
                          rows=rows)
    save('training_audit.json', dict(datasets=audit, source_sha256=hashes,
        limitations=['Historical failed confirmation now explicitly used for mechanism development',
                     'Rank is not correctness; no causal attribution for individual errors',
                     'No logits saved: soft2 outcomes cannot be replayed from these traces',
                     'No full-test prediction files were read by this audit']))
    save('development_usage.json', usage)
    peer = Path('E:/srtp/srtp-final')
    plan = read(peer/'integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json')
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(s) for s in train_path.read_text(encoding='utf-8').splitlines()]
    excluded_ids = set().union(*(set(v) for v in plan['exclusions'].values()))
    for stages in plan['splits'].values():
        for ids in stages.values():
            excluded_ids.update(ids)
    blocked = {phash(train[i]['problem']) for i in excluded_ids}
    scanned = {}; errors = []

    def inspect(obj):
        if isinstance(obj, dict):
            for k in ('problem', 'question'):
                if isinstance(obj.get(k), str):
                    blocked.add(phash(obj[k]))
            for k in ('problem_sha256', 'normalized_prompt_sha256'):
                if isinstance(obj.get(k), str) and len(obj[k]) == 64:
                    blocked.add(obj[k])
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    inspect(v)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, (dict, list)):
                    inspect(v)

    for base in (ROOT, peer):
        folders = [base/'integration/rebalance_easysteer/configs',
                   base/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912']
        for folder in folders:
            for p in folder.rglob('*'):
                if (not p.is_file() or p.suffix not in ('.json', '.jsonl')
                        or HERE in p.parents or p.name == 'gsm8k_source_train.jsonl'):
                    continue
                raw = p.read_bytes(); scanned[str(p)] = hashlib.sha256(raw).hexdigest()
                try:
                    if p.suffix == '.json':
                        inspect(json.loads(raw.decode('utf-8-sig')))
                    else:
                        for s in raw.decode('utf-8-sig').splitlines():
                            if s.strip(): inspect(json.loads(s))
                except (ValueError, UnicodeError) as e:
                    errors.append(dict(path=str(p), error=str(e)))
    assert not errors, errors
    # Test question text is used solely for exclusion, never test predictions.
    for name in ('Math_Math500', 'Math_GSM8K'):
        p = ROOT/f'sources/ReBalance/Data/{name}/test.jsonl'
        scanned[str(p)] = sha(p)
        for s in p.read_text(encoding='utf-8').splitlines(): inspect(json.loads(s))
    gsm_path = ns/'expanded_20260913/gsm8k_source_train.jsonl'
    gsm = [json.loads(s) for s in gsm_path.read_text(encoding='utf-8').splitlines()]
    selected = {}; available = {}
    for role, source_rows in [('math_soft2', train), ('gsm8k_soft2', gsm)]:
        eligible = []; seen = set(blocked)
        for i, r in enumerate(source_rows):
            problem = r.get('problem', r.get('question'))
            h = phash(problem)
            if h in seen: continue
            seen.add(h)
            answer = r['answer'] if role.startswith('math') else r['answer'].split('####')[-1].strip()
            if answer is None:
                continue  # Unusable pre-existing reference, independent of model output.
            eligible.append(dict(r, dataset='math_train' if role.startswith('math') else 'gsm8k_train',
                                 train_index=i, problem=problem, answer=answer, problem_sha256=h))
        eligible.sort(key=lambda r: hashlib.sha256(
            ('hybrid_syncthink_cgrs_20260912|soft2_screen_v1|' + r['problem_sha256']).encode()).hexdigest())
        assert len(eligible) >= 64
        selected[role] = eligible[:64]; available[role] = len(eligible)
        blocked.update(r['problem_sha256'] for r in selected[role])
        save(role+'.json', selected[role])
    save('data_proposal.json', dict(
        status='proposed_not_frozen_pending_executor_global_reconciliation',
        normalization='Unicode NFKC then remove all Unicode whitespace; SHA256 UTF-8',
        peer_commit=subprocess.check_output(['git','-C',str(peer),'rev-parse','HEAD'],text=True).strip(),
        scanned_sha256=scanned, parse_errors=errors, eligible=available,
        source_sha256={str(train_path):sha(train_path), str(gsm_path):sha(gsm_path)},
        question_files={role:sha(HERE/(role+'.json')) for role in selected},
        proposals=[dict(dataset=r['dataset'], train_index=r['train_index'],
                        problem_sha256=r['problem_sha256'], purpose='soft2_screening',
                        status='proposed_not_frozen') for group in selected.values() for r in group],
        limitations=['Local tracked data/config snapshot only; remote/private claims require executor check',
                     'No confirmation questions reserved; no A/B registry or peer file modified']))
    print(json.dumps(dict(audit={k:{n:v for n,v in d.items() if n!='rows'} for k,d in audit.items()},
                         proposed=128, scanned=len(scanned), eligible=available), indent=2))


if __name__ == '__main__':
    main()
