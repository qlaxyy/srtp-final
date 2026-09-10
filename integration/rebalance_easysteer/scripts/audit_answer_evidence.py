"""Index saved training answers and validate reviewer-supplied answer evidence.

CPU/stdlib only. Candidate markers and exact repeats are retrieval aids, never
overthinking labels. No model calls, new generation, or automatic semantic judge.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from audit_calibration_labels import ROOT, DEFAULT, read, save, sha, decoder

OUTPUT = ROOT / '.codex_work/label_evidence_30_20260910/v2'
AUDIT = ROOT / 'integration/rebalance_easysteer/configs/calibration_label_audit30_20260910.json'
MARKER = re.compile(r'\\boxed|\\fbox|\b(answer|therefore|thus|hence|solution|domain|range|probability|minimum|maximum|cost|ways|total|zeros|result)\b', re.I)


def rational_literal(value):
    """Parse only exact scalar literals; never search a longer string for a number."""
    value = value.strip().replace('\u2212', '-')
    match = re.fullmatch(r'([+-]?)\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', value)
    try:
        if match:
            return (-1 if match[1] == '-' else 1) * Fraction(int(match[2]), int(match[3]))
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', value):
            if '/' in value:
                top, bottom = value.split('/')
                return Fraction(top) / Fraction(bottom)
            return Fraction(value)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def compare_literals(answer, gold):
    a, g = rational_literal(answer), rational_literal(gold)
    if a is not None and g is not None:
        return 'correct' if a == g else 'incorrect'
    return 'unverified'


def prepare(audit_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evidence_index.json').exists():
        raise FileExistsError('Evidence index already exists; reuse it, do not overwrite')
    protocol = read(audit_dir / 'audit_protocol.json')
    assert sha(audit_dir / 'private_key.json') == protocol['private_key_sha256']
    assert sha(audit_dir / 'tokenizer.json') == protocol['tokenizer_sha256']
    decode, vocab = decoder(read(audit_dir / 'tokenizer.json'))
    boundary = {i for text, i in vocab.items() if '\u010a\u010a' in text}
    frozen_path = ROOT / 'integration/rebalance_easysteer/configs/final_results_20260909.json'
    frozen = read(frozen_path)
    assert boundary == set(frozen['benchmarks'][0]['protocol']['dynamic_params']['boundary_token_ids'])
    end_id = frozen['benchmarks'][0]['protocol']['dynamic_params']['think_end_token_id']
    old = read(audit_dir / 'private_questions.json')
    by_index = {q['calibration_index']: q for q in old.values()}
    previous = {r['case_id']: r for r in read(AUDIT)['reviewed_cases']}
    train_path = ROOT / 'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line]
    questions = []
    archive_path = ROOT / protocol['source_archive']
    with tarfile.open(archive_path) as archive:
        with archive.extractfile(protocol['source_member']) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == protocol['source_sha256']
        for index, line in enumerate(archive.extractfile(protocol['source_member'])):
            if index not in by_index:
                continue
            row, saved = json.loads(line), by_index[index]
            gold = train[row['train_index']]
            assert row['train_index'] == saved['train_index']
            assert row['problem'] == saved['problem'] == gold['problem']
            ids = row['token_ids']
            assert len(ids) <= protocol['source_max_tokens'] == 16000
            closed = end_id in ids
            end = ids.index(end_id) if closed else len(ids)
            first, steps, seen = 0, [], {}
            for stop in range(end + 1):
                if not (stop < end and ids[stop] in boundary) and not (stop == end and closed):
                    continue
                if stop > first:
                    content = decode(ids[first:stop])
                    separator = decode(ids[stop:stop + 1]) if stop < end else ''
                    step = dict(step=len(steps) + 1, start=first, stop=stop, text=content,
                                boundary_text=separator, evidence_text=content + separator,
                                evidence_stop=stop + (stop < end),
                                retrieval_marker=bool(MARKER.search(content + separator)),
                                exact_repeat_of=seen.get(content))
                    steps.append(step)
                    seen.setdefault(content, step['step'])
                first = stop + 1
            assert [{'step': s['step'], 'text': s['text']} for s in steps] == saved['steps']
            targets = []
            for target in saved['targets']:
                historical = previous[target['case_id']]
                targets.append(dict(**target, historical_review=historical.get('review', historical)))
            questions.append(dict(question=saved['question'], calibration_index=index,
                train_index=row['train_index'], problem=row['problem'], gold_answer=gold['answer'],
                reference_solution=gold['solution'], output_tokens=len(ids),
                finish_reason=row['finish_reason'], thinking_closed=closed,
                steps=steps, targets=targets,
                unfinished_reasoning_tail=decode(ids[first:end]) if not closed else '',
                final_answer_text=decode(ids[end + 1:], skip_special=True) if closed else ''))
    assert len(questions) == protocol['question_count'] == 30
    payload = dict(protocol=dict(source_sha256=protocol['source_sha256'],
        source_archive=protocol['source_archive'], source_member=protocol['source_member'],
        source_model=protocol['source_model'], source_temperature=protocol['source_temperature'],
        source_max_tokens=protocol['source_max_tokens'], sample_seed=protocol['seed'],
        prior_audit_sha256=sha(AUDIT), training_data_sha256=sha(train_path),
        new_generations=0, model_forwards=0, gpu_runs=0,
        token_offsets='0-based half-open; start:stop is frozen calibration span; start:evidence_stop restores boundary punctuation for reading',
        marker_role='retrieval only; not an answer extractor or semantic label',
        split='same 30 already-exposed training questions; development audit, not held-out validation'),
        questions=questions)
    save(output / 'evidence_index.json', payload)
    print(json.dumps({'questions': len(questions), 'checkpoints': sum(len(q['targets']) for q in questions),
        'complete_steps': sum(len(q['steps']) for q in questions),
        'capped': sum(q['finish_reason'] == 'length' for q in questions)}, ensure_ascii=False))


def show(output, first, last, all_steps=False):
    review_path = output / 'answer_anchor_review.json'
    reviews = {r['question']: r for r in read(review_path)} if review_path.exists() else {}
    for q in read(output / 'evidence_index.json')['questions']:
        if not first <= int(q['question'][1:]) <= last:
            continue
        print('\n' + q['question'], 'train', q['train_index'], 'GOLD', q['gold_answer'], 'CAPPED', q['finish_reason'])
        print('PROBLEM:', q['problem'])
        print('CHECKPOINTS:', json.dumps(q['targets'], ensure_ascii=False))
        anchors = [validate_anchor(a, q) for a in reviews.get(q['question'], {}).get('anchors', [])]
        reviewed_steps = {a['step'] for a in anchors}
        if anchors:
            print('REVIEWED ANSWER EVIDENCE:', json.dumps(anchors, ensure_ascii=False))
        targets = {t['step'] for t in q['targets']}
        for step in q['steps']:
            if step['exact_repeat_of'] and step['step'] not in targets | reviewed_steps:
                continue
            near = any(abs(step['step'] - t) <= 1 for t in targets)
            if all_steps or step['retrieval_marker'] or near or step['step'] in reviewed_steps:
                print(f"[{step['step']}] {step['evidence_text'].rstrip()}")
        for name, text in [('UNFINISHED TAIL', q['unfinished_reasoning_tail']),
                           ('FINAL ANSWER', q['final_answer_text'])]:
            print(name + ':', text[:800])
            if len(text) > 800:
                print(f'[display limited to 800/{len(text)} characters; full text retained in index]')


def validate_anchor(anchor, question):
    assert 1 <= anchor['step'] <= len(question['steps'])
    step = question['steps'][anchor['step'] - 1]
    assert step['step'] == anchor['step']
    assert anchor['quote'] and anchor['quote'] in step['evidence_text']
    assert anchor['scope'] in ['original_problem', 'intermediate', 'retracted_or_hypothetical']
    assert anchor['verdict'] in ['correct', 'incorrect', 'uncertain']
    assert anchor['basis']
    assert all(1 <= n <= anchor['step'] for n in anchor.get('context_steps', []))
    automatic = compare_literals(anchor['answer'], question['gold_answer'])
    if anchor['scope'] == 'original_problem' and automatic != 'unverified':
        assert anchor['verdict'] == automatic, (question['question'], anchor, automatic)
    position = step['evidence_text'].index(anchor['quote'])
    return dict(**anchor, start=step['start'], stop=step['stop'], evidence_stop=step['evidence_stop'],
        quote_char_start=position, quote_char_stop=position + len(anchor['quote']),
        literal_check=automatic if anchor['scope'] == 'original_problem' else 'not_applicable')


def prefix_evidence(anchors, step):
    # A future claim must never be used to label an earlier checkpoint.
    observed = [a for a in anchors if a['step'] <= step and a['scope'] == 'original_problem']
    verdicts = {a['verdict'] for a in observed}
    state = ('conflicting_observed_claims' if {'correct', 'incorrect'} <= verdicts else
             'correct_claim_observed' if 'correct' in verdicts else
             'incorrect_claim_observed' if 'incorrect' in verdicts else
             'uncertain_claim_observed' if observed else 'no_reviewed_answer_evidence')
    return dict(answer_evidence=state, reviewed_answer_steps=[a['step'] for a in observed],
                latest_reviewed_answer_step=max((a['step'] for a in observed), default=None),
                latest_reviewed_answer_verdict=(max(observed, key=lambda a: a['step'])['verdict'] if observed else None))


def finalize(output, review_path, report_path):
    index, reviews = read(output / 'evidence_index.json'), read(review_path)
    by_q = {r['question']: r for r in reviews}
    assert len(by_q) == len(reviews) == len(index['questions']) == 30
    questions, checkpoints = [], []
    for q in index['questions']:
        review = by_q[q['question']]
        anchors = sorted([validate_anchor(a, q) for a in review['anchors']], key=lambda a: a['step'])
        assert review['coverage_note']
        updates = {p['case_id']: p for p in review.get('progress_updates', [])}
        assert len(updates) == len(review.get('progress_updates', []))
        assert set(updates) <= {t['case_id'] for t in q['targets']}
        points = []
        for target in q['targets']:
            step = q['steps'][target['step'] - 1]
            historical = target['historical_review']
            current = updates.get(target['case_id'], historical)
            assert current['evidence_quote'] in step['text']
            assert current['category'] in ['new', 'repeat', 'uncertain']
            assert current['math_validity'] in ['supported', 'error', 'uncertain', 'not_applicable']
            prior = current['prior_steps']
            assert all(1 <= n < step['step'] for n in prior)
            point = dict(question=q['question'], train_index=q['train_index'],
                case_id=target['case_id'], step=step['step'], start=step['start'], stop=step['stop'],
                **prefix_evidence(anchors, step['step']),
                progress_evidence=current['category'], math_validity=current['math_validity'],
                progress_quote=current['evidence_quote'], prior_steps=prior,
                progress_reason=current['reason'],
                prior_evidence=[dict(step=n, text=q['steps'][n - 1]['evidence_text']) for n in prior],
                progress_review_source=('current context amendment' if target['case_id'] in updates else
                    'previous 76-step content audit; evidence text and prefix references rechecked'),
                historical_review=historical,
                amendment_basis=current.get('change_basis'),
                thinking_class='not_assigned', causal_sufficiency='not_tested')
            points.append(point)
            checkpoints.append(point)
        questions.append(dict(question=q['question'], train_index=q['train_index'],
            calibration_index=q['calibration_index'], gold_answer=q['gold_answer'],
            finish_reason=q['finish_reason'], output_tokens=q['output_tokens'],
            thinking_closed=q['thinking_closed'],
            incomplete_tail_characters=len(q['unfinished_reasoning_tail']),
            answer_anchors=anchors, coverage_note=review['coverage_note'],
            annotation_input=review, checkpoints=points))
    states = Counter(p['answer_evidence'] for p in checkpoints)
    progress = {k: dict(Counter(p['answer_evidence'] for p in checkpoints if p['progress_evidence'] == k))
                for k in ['new', 'repeat', 'uncertain']}
    result = dict(status='completed_cpu_evidence_index_and_single_reviewer_anchor_audit',
        date='2026-09-10', execution_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        purpose='Organize observable answer claims and progress evidence, not relabel over/under/normal',
        protocol=index['protocol'], question_count=len(questions), checkpoint_count=len(checkpoints),
        capped_questions=sum(q['finish_reason'] == 'length' for q in questions),
        answer_evidence_counts=dict(states), progress_by_answer_evidence=progress,
        review_source='Codex-assisted manual semantic selection; exact rational checks where supported; no independent human gold',
        limitations=['Only reviewed explicit answer claims are counted; no claim of finding the earliest answer or latent readiness.',
            'No reviewed answer evidence does not mean no answer exists or underthinking.',
            'conflicting_observed_claims is historical evidence of both right and wrong claims, not a claim that the current state is unresolved.',
            'A correct claim does not prove the next forced answer would be correct, or that steering is safe.',
            'Incomplete tails and final answers retained in local index; anchors only use complete reasoning steps.',
            'Historical content labels are reused with reference checks, not presented as an independent new semantic assessment.',
            'Selected strata are uneven and this is a development sample; no population error rate.'],
        artifacts=dict(index_path=str((output / 'evidence_index.json').relative_to(ROOT)),
            index_sha256=sha(output / 'evidence_index.json'), review_sha256=sha(review_path),
            script_sha256=sha(__file__)), questions=questions)
    save(report_path, result)
    print(json.dumps({'questions': len(questions), 'checkpoints': len(checkpoints),
        'answer_evidence_counts': dict(states), 'progress_by_answer_evidence': progress}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'show', 'finalize'])
    parser.add_argument('--audit-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, default=30)
    parser.add_argument('--all-steps', action='store_true')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.audit_dir, args.output)
    elif args.command == 'show':
        show(args.output, args.first, args.last, args.all_steps)
    else:
        if not args.review or not args.report:
            parser.error('finalize requires --review and --report')
        finalize(args.output, args.review, args.report)
