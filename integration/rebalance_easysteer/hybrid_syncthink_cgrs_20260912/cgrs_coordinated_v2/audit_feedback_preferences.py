"""Read-only CPU audit of full-response preferences; does not fit a vector."""
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    home = Path(__file__).resolve().parent
    root = next(p for p in home.parents if (p / '.codex_work').is_dir())
    old = home / 'self_feedback_train500_20260918_run1'
    endpoints = home / 'outcome_endpoints_20260918_run1'
    out = home / 'feedback_preference_audit_20260918_run1'
    out.mkdir(exist_ok=False)
    labels_path = old / 'labels.json'
    archive = root / '.codex_work/self_feedback_train500_20260918_evidence.tar.gz'
    assert sha(labels_path) == '194fcb92205dd3a7275be0cbd9116ffa028d8ec57e64a245c826fbda1a5a1869'
    assert sha(archive) == 'cd38a0f003ed26d53037dbc9c74483e1943d8032bf150ae237e8cdb8c2b443ca'
    labels = json.loads(labels_path.read_text())
    plan = json.loads((old / 'plan.json').read_text())
    pairs = json.loads((endpoints / 'pairs.json').read_text())
    audit = json.loads((endpoints / 'short_error_audit.json').read_text())
    with tarfile.open(archive) as tf:
        l27 = json.load(tf.extractfile('self_feedback_train500_20260918_run1/merged/L27_L27/result.json'))['records']
    sources = {'U': json.loads((old / 'baseline.json').read_text()), 'L27': l27}
    records, inversions = [], []
    for pair in pairs:
        q = pair['question']
        if pair['kind'] == 'CW' and q not in audit['remaining']:
            continue
        long, short = pair['long_source'], pair['short_source']
        if pair['kind'] == 'CC':
            chosen, rejected = short, long
            kind = 'correct_short_over_correct_long'
        else:
            chosen, rejected = long, short
            kind = 'correct_long_over_wrong_short'
        row = dict(question=q, train_index=sources[chosen][q]['train_index'],
                   problem_sha256=pair['problem_sha256'], kind=kind,
                   chosen_source=chosen, rejected_source=rejected,
                   prompt_tokens=len(plan['rows'][q]['prompt_token_ids']), endpoints={})
        for role, source in [('chosen', chosen), ('rejected', rejected)]:
            rec, lab = sources[source][q], labels[q][source]
            assert rec['problem_sha256'] == pair['problem_sha256'] == lab['problem_sha256']
            assert hashlib.sha256(rec['text'].encode()).hexdigest() == lab['text_sha256']
            ids = rec['token_ids']
            assert len(ids) == lab['tokens'] and ids.index(151649) == lab['thinking_tokens']
            assert lab['closed'] and lab['finish_reason'] != 'length'
            row['endpoints'][role] = dict(source=source, correct=lab['correct'],
                thinking_tokens=lab['thinking_tokens'], total_tokens=len(ids),
                prompt_plus_response_tokens=row['prompt_tokens']+len(ids),
                text_sha256=lab['text_sha256'],
                token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                last_token_id=ids[-1])
        if pair['kind'] == 'CC' and row['endpoints']['chosen']['total_tokens'] >= row['endpoints']['rejected']['total_tokens']:
            inversions.append(q)
        records.append(row)
    assert len(records) == 236 and len({r['problem_sha256'] for r in records}) == 236
    cc = [r for r in records if r['kind'] == 'correct_short_over_correct_long']
    cw = [r for r in records if r['kind'] == 'correct_long_over_wrong_short']
    def stats(rows):
        return dict(parents=len(rows), chosen_sources=dict(Counter(r['chosen_source'] for r in rows)),
            response_tokens=sum(e['total_tokens'] for r in rows for e in r['endpoints'].values()),
            prompt_plus_response_tokens=sum(e['prompt_plus_response_tokens'] for r in rows for e in r['endpoints'].values()),
            max_prompt_plus_response=max(e['prompt_plus_response_tokens'] for r in rows for e in r['endpoints'].values()),
            pairs_exceeding_context={str(n):sum(any(e['prompt_plus_response_tokens']>n for e in r['endpoints'].values()) for r in rows) for n in [2048,4096,8192,16384]},
            mean_thinking_gap=sum(r['endpoints']['rejected']['thinking_tokens']-r['endpoints']['chosen']['thinking_tokens'] for r in rows)/len(rows))
    summary = dict(status='CPU only; no new GPU, trained vector, benchmark, or independent split',
        correctness_first=stats(cw), compression=stats(cc),
        compression_pairs_not_improving_total_tokens=inversions,
        author_false_negative_questions_excluded=audit['excluded'],
        source_hashes={str(p.relative_to(root)):sha(p) for p in [labels_path, archive, old/'baseline.json', old/'plan.json', endpoints/'pairs.json', endpoints/'short_error_audit.json', Path(__file__)]},
        limitations=['Existing outcome labels are final-answer labels, not step correctness or causal overthinking labels.',
            'All parents have been used in prior research; any future split is development-only, not fresh independent confirmation.',
            'The five correctness pairs are a diagnostic guard, insufficient to certify a 2 percentage-point accuracy bound.',
            'Context overflow audit assumes preserving the full prompt; it is not an execution of the official trainer truncation code.',
            'Token count includes saved final answer; terminal stop handling must be checked before a likelihood objective.',
            'No confidence or variance filter is applied; no class is interpreted as the midpoint of efficient reasoning.'])
    save(out/'parents.json', records)
    save(out/'summary.json', summary)
    (out/'.gitattributes').write_text('*.json -text\n', encoding='ascii')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
