"""CPU-only preparation and unmasking of a saved-calibration content audit.

No model execution, new generations, automatic semantic judging, or fit changes.
The reviewer writes judgments separately before running ``unmask``.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tarfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / '.codex_work/label_audit_30_20260910'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lexical_rule():
    path = ROOT / 'sources/ReBalance/hidden_analysis_auto.py'
    names = {'LEXICON_BASE', 'LEXICON_RE', '_normalize_text', '_token_pattern',
             '_compile_lexicon_regex', 'has_lexicon_hit'}
    selected = []
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    namespace = {'re': re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['has_lexicon_hit'], namespace['LEXICON_RE']


def decoder(tokenizer):
    # The frozen tokenizer uses the standard ByteLevel decoder, with added tokens.
    assert tokenizer['decoder']['type'] == 'ByteLevel'
    byte_values = list(range(ord('!'), ord('~') + 1)) + list(range(161, 173)) + list(range(174, 256))
    chars = list(byte_values)
    extra = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            chars.append(256 + extra)
            extra += 1
    inverse = {chr(c): b for c, b in zip(chars, byte_values)}
    vocab = tokenizer['model']['vocab']
    raw = {i: bytes(inverse[c] for c in token) for token, i in vocab.items()}
    special = set()
    for token in tokenizer['added_tokens']:
        raw[token['id']] = token['content'].encode('utf-8')
        if token['special']:
            special.add(token['id'])

    def decode(ids, skip_special=False):
        return b''.join(raw[i] for i in ids if not skip_special or i not in special).decode('utf-8', errors='replace')
    return decode, vocab


def quantile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'audit_protocol.json').exists():
        raise FileExistsError('Audit sample already fixed; do not resample')
    frozen_path = ROOT / 'integration/rebalance_easysteer/configs/final_results_20260909.json'
    frozen = read(frozen_path)
    calibration_path = ROOT / '.codex_work/auto_code_v2_500_20260908/protocol.json'
    calibration = read(calibration_path)
    assert sha(out / 'tokenizer.json') == frozen['assets']['models']['1.5B']['tokenizer.json']['sha256']
    decode, vocab = decoder(read(out / 'tokenizer.json'))
    boundary = {i for t, i in vocab.items() if 'ĊĊ' in t}
    params = frozen['benchmarks'][0]['protocol']['dynamic_params']
    assert boundary == set(params['boundary_token_ids'])
    end_id = params['think_end_token_id']
    is_lexical, regex = lexical_rule()
    cl, ch = calibration['confidence_quantiles']
    seed = 20260910
    selected = sorted(random.Random(seed).sample(range(500), 30))
    aliases = {q: f'Q{i + 1:02d}' for i, q in enumerate(selected)}
    all_c, all_v, question_cases, key = [], [], {}, []
    counts = Counter()
    decode_tail_holds = []
    archive_path = ROOT / '.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz'
    test_questions = set()
    for name in ['Math_Math500', 'Math_GSM8K']:
        path = ROOT / 'sources/ReBalance/Data' / name / 'test.jsonl'
        test_questions.update(''.join(json.loads(line)['problem'].split())
                              for line in path.read_text(encoding='utf-8').splitlines() if line)
    with tarfile.open(archive_path) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        assert manifest['temperature'] == 0 and manifest['count'] == 500
        with archive.extractfile('generations.jsonl') as stream:
            source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert source_sha == calibration['source_sha256']
        for q, line in enumerate(archive.extractfile('generations.jsonl')):
            row = json.loads(line)
            assert row['train_index'] == manifest['train_indices'][q]
            assert ''.join(row['problem'].split()) not in test_questions
            ids, logs = row['token_ids'], row['logprobs']
            assert len(ids) == len(logs) <= 16000
            assert all(math.isfinite(p) and p <= 1e-5 for p in logs)
            decoded = decode(ids, skip_special=True)
            if decoded != row['text']:
                # vLLM may withhold a final token containing incomplete UTF-8.
                # Such a capped, unfinished tail is excluded by the step policy.
                suffix = decoded[len(row['text']):]
                assert (row['finish_reason'] == 'length' and len(ids) == 16000
                        and decoded.startswith(row['text']) and '\ufffd' in suffix
                        and all(c.isspace() or c == '\ufffd' for c in suffix)), q
                decode_tail_holds.append(dict(calibration_index=q, suffix_codepoints=[ord(c) for c in suffix]))
            closed = end_id in ids
            end = ids.index(end_id) if closed else len(ids)
            first, previous, steps = 0, None, []
            for stop in range(end + 1):
                if not (stop < end and ids[stop] in boundary) and not (stop == end and closed):
                    continue
                if stop > first:
                    c = math.fsum(math.exp(p) for p in logs[first:stop]) / (stop - first)
                    v = 0.0 if previous is None else (c - previous) ** 2 / 4
                    text = decode(ids[first:stop])
                    lexical = bool(is_lexical(text))
                    label = 'over' if lexical or c < cl else 'under' if c > ch else 'unselected'
                    route = ('both' if lexical and c < cl else 'lexical_only' if lexical else
                             'low_confidence_only' if c < cl else label)
                    step = dict(step=len(steps) + 1, start=first, stop=stop, text=text,
                                confidence=c, variance=v, lexical_hit=lexical,
                                lexical_matches=[m.group() for m in regex.finditer(text)],
                                original_label=label, route=route)
                    steps.append(step)
                    all_c.append(c)
                    all_v.append(v)
                    counts[label] += 1
                    previous = c
                first = stop + 1
            if q not in aliases:
                continue
            targets = []
            available = {}
            for label in ['over', 'under', 'unselected']:
                eligible = [s for s in steps if s['original_label'] == label]
                available[label] = len(eligible)
                if not eligible:
                    continue
                target = random.Random(f'{seed}:{q}:{label}').choice(eligible)
                case_id = 'S' + hashlib.sha256(f'{seed}:{q}:{target["step"]}'.encode()).hexdigest()[:8]
                targets.append(dict(case_id=case_id, step=target['step']))
                key.append(dict(case_id=case_id, question=aliases[q], calibration_index=q,
                                train_index=row['train_index'], **target))
            targets.sort(key=lambda s: s['step'])
            question_cases[aliases[q]] = dict(question=aliases[q], calibration_index=q,
                train_index=row['train_index'], problem=row['problem'],
                targets=targets, steps=[dict(step=s['step'], text=s['text']) for s in steps],
                available=available, capped=row['finish_reason'] == 'length', tokens=len(ids))
    assert q + 1 == 500 and len(all_c) == calibration['steps'] == 84008
    fit = read(calibration_path.with_name('fit.json'))
    assert counts['over'] == fit['positives'] and counts['under'] == fit['negatives']
    for values, expected in [(all_c, calibration['confidence_quantiles']),
                              (all_v, calibration['variance_quantiles'])]:
        assert max(abs(quantile(values, p) - e) for p, e in zip([.25, .75], expected)) < 1e-12
    save(out / 'private_key.json', key)
    save(out / 'private_questions.json', question_cases)
    blind_dir = out / 'blind'
    blind_dir.mkdir(exist_ok=True)
    for question, record in question_cases.items():
        previous = 0
        for round_index, target in enumerate(record['targets']):
            end_step = target['step']
            # Each packet reveals only the prefix through its current target.
            # Later packets contain just the new span; earlier packets remain available.
            body = dict(question=question, case_id=target['case_id'], target_step=end_step,
                        problem=record['problem'], prior_packet_count=round_index,
                        context_from_step=previous + 1,
                        steps=record['steps'][previous:end_step])
            save(blind_dir / f'{round_index + 1}_{question}.json', body)
            previous = end_step
    protocol = dict(status='prepared_unreviewed', seed=seed, question_count=30,
        target_count=len(key), max_targets_per_question=3, sampling='uniform questions; one random step per available original stratum per question; no replacement for absent strata',
        strata=['over', 'under', 'unselected'], selected_calibration_indices=selected,
        source_archive=str(archive_path.relative_to(ROOT)), source_member='generations.jsonl',
        source_sha256=source_sha, source_model=calibration['model'], source_count=500,
        source_temperature=manifest['temperature'], source_max_tokens=manifest['max_tokens'],
        regenerated_answers=0, model_forwards=0, server_compute=False, population_steps=len(all_c),
        population_label_counts=dict(counts), full_decode_verified_questions=500,
        capped_incomplete_utf8_tail_holds=decode_tail_holds,
        current_quantiles_reproduced=True, test_overlap=0,
        tokenizer_sha256=sha(out / 'tokenizer.json'), frozen_manifest_sha256=sha(frozen_path),
        private_key_sha256=sha(out / 'private_key.json'),
        rubric=dict(new='New derivation, condition, case elimination, correction or different verification. Mathematical validity recorded separately.',
                    repeat='Equivalent prior reasoning or conclusion without new condition, calculation or verification. Must cite an earlier step.',
                    uncertain='Mixed, transitional, or insufficient evidence.'),
        validity_values=['supported', 'error', 'uncertain', 'not_applicable'],
        reviewer='Single Codex model-assisted review; no independent human gold labels',
        masking='Original label, confidence, match flags and final grading hidden. Review packets in numerical rounds; seal each round before reading later prefixes. Visible wording can suggest the heuristic label; not perfect blindness.',
        interpretation='Content audit only, not causal redundancy, underthinking ground truth, population error rate or efficacy evidence. Stratified counts must not be pooled as prevalence.',
        incidental_prior_exposure='Schema inspection printed calibration question 0 before sampling. No other selected answer was read before preparation.',
        preexposed_selected_questions=[aliases[0]] if 0 in aliases else [])
    save(out / 'audit_protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ['question_count', 'target_count', 'selected_calibration_indices',
          'full_decode_verified_questions', 'preexposed_selected_questions']}, ensure_ascii=False))
    print('Blind packet characters:', sum(p.stat().st_size for p in blind_dir.glob('*.json')))


def show(out, round_index, first, last):
    for q in range(first, last + 1):
        path = out / 'blind' / f'{round_index}_Q{q:02d}.json'
        if not path.exists():
            continue
        packet = read(path)
        print(f'\n{packet["question"]} {packet["case_id"]} TARGET STEP {packet["target_step"]}')
        print('PROBLEM:', packet['problem'])
        print('CONTEXT (earlier rounds remain part of prefix):')
        seen = {}
        for earlier in range(1, round_index):
            for step in read(out / 'blind' / f'{earlier}_Q{q:02d}.json')['steps']:
                seen.setdefault(step['text'], step['step'])
        repeated = []
        for step in packet['steps']:
            mark = ' TARGET' if step['step'] == packet['target_step'] else ''
            if step['text'] in seen and not mark:
                repeated.append(f'{step["step"]}={seen[step["text"]]}')
                continue
            if repeated:
                print('EXACT REPEAT REFERENCES (current=earlier):', ', '.join(repeated))
                repeated = []
            print(f'[{step["step"]}{mark}] {step["text"]}')
            seen.setdefault(step['text'], step['step'])


def unmask(out):
    protocol = read(out / 'audit_protocol.json')
    assert sha(out / 'private_key.json') == protocol['private_key_sha256']
    judgments, seals = [], []
    for round_index in [1, 2, 3]:
        path = out / f'judgments_round{round_index}.json'
        rows = read(path)
        expected = {read(p)['case_id'] for p in (out / 'blind').glob(f'{round_index}_*.json')}
        assert len(rows) == len(expected) and {r['case_id'] for r in rows} == expected
        for row in rows:
            assert row['category'] in protocol['rubric']
            assert row['math_validity'] in protocol['validity_values']
            assert row['reason'] and row['evidence_quote']
            packet_path = next(p for p in (out / 'blind').glob(f'{round_index}_*.json') if read(p)['case_id'] == row['case_id'])
            packet = read(packet_path)
            assert row['evidence_quote'] in packet['steps'][-1]['text']
            if row['category'] == 'repeat':
                assert row['prior_steps'] and max(row['prior_steps']) < packet['target_step']
            judgments.append(row)
        seals.append(dict(round=round_index, path=path.name, sha256=sha(path)))
    by_id = {row['case_id']: row for row in judgments}
    key = read(out / 'private_key.json')
    assert len(key) == len(by_id) == protocol['target_count']
    joined = [dict(**row, review=by_id[row['case_id']]) for row in key]
    matrix = {label: dict(Counter(r['review']['category'] for r in joined if r['original_label'] == label))
              for label in protocol['strata']}
    routes = {route: dict(Counter(r['review']['category'] for r in joined if r['route'] == route))
              for route in sorted({r['route'] for r in joined})}
    summary = dict(status='completed_single_reviewer_content_audit', protocol=protocol,
        judgment_seals=seals, by_original_label=matrix, by_trigger_route=routes,
        questions=30, reviewed_steps=len(joined), cases=joined)
    save(out / 'unmasked_audit.json', summary)
    print(json.dumps(dict(by_original_label=matrix, by_trigger_route=routes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'show', 'unmask'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--round', type=int, default=1, dest='round_index')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, default=30)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'show':
        show(args.output, args.round_index, args.first, args.last)
    else:
        unmask(args.output)
