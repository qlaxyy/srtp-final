"""Find formatting duplicates and close text pairs without reading model outputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
import unicodedata

from mechanism_candidates import ROOT, read, save, sha, require


def canonical(text):
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = re.sub(r'\\(?:left|right)\b', '', text)
    text = re.sub(r'\\(?:quad|qquad)\b|\\[,;! ]', '', text)
    return re.sub(r'\s+', '', text.replace('$', '')).rstrip('.')


def audit(bundle):
    started = time.perf_counter(); plan = read(bundle/'plan.json')
    source = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    require(sha(source, source=True) == plan['train_sha256'], 'Training source changed')
    train = [json.loads(line)['problem'] for line in source.read_text(encoding='utf-8').splitlines()]
    selected = sorted({i for stages in plan['splits'].values() for ids in stages.values() for i in ids})
    excluded = set().union(*(set(ids) for ids in plan['exclusions'].values()))
    require(len(selected) == 608 and not excluded.intersection(selected), 'Wrong planned split')
    pool = {f'train:{i}': train[i] for i in sorted(excluded | set(selected))}
    for relative, digest in plan['test_prompt_source_sha256'].items():
        path = ROOT/relative; require(sha(path, source=True) == digest, 'Test source changed')
        name = path.parent.name
        for i, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            pool[f'{name}:{i}'] = json.loads(line)['problem']
    forms = {name: canonical(text) for name, text in pool.items()}
    grams = {name: {value[i:i+5] for i in range(max(1, len(value)-4))}
             for name, value in forms.items()}
    inverted = defaultdict(list)
    for name, values in grams.items():
        for gram in values:
            inverted[gram].append(name)
    exact = []; near = []; comparisons = 0; seen = set()
    for index in selected:
        name = f'train:{index}'; values = grams[name]; shared = Counter()
        for gram in values:
            shared.update(inverted[gram])
        for other, common in shared.items():
            key = tuple(sorted([name, other]))
            if other == name or key in seen:
                continue
            seen.add(key); comparisons += 1
            score = common/(len(values)+len(grams[other])-common)
            if forms[name] == forms[other] or score >= .88:
                item = dict(planned_train_index=index, compared_id=other,
                    canonical_equal=forms[name] == forms[other], character_5gram_jaccard=score,
                    planned_problem=pool[name], compared_problem=pool[other])
                (exact if item['canonical_equal'] else near).append(item)
    near.sort(key=lambda row: -row['character_5gram_jaccard'])
    return dict(status='CPU_prompt_similarity_screen_requires_manual_duplicate_judgment',
        parent_plan_sha256=sha(bundle/'plan.json'), source_sha256=sha(Path(__file__), source=True),
        selected_questions=608, historical_and_test_and_selected_pool=len(pool),
        comparisons_with_any_shared_gram=comparisons, canonical_matches=exact, near_matches=near,
        decision='No automatic split mutation. Review flagged pairs for same question with the same values; similar templates with different values remain different questions.',
        limits='Formatting and character similarity only; absence of a match does not prove semantic independence or absence from model pretraining. No model outputs or correctness labels were used.',
        cpu_seconds=time.perf_counter()-started, GPU_calls=0, new_answers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Immutable audit exists')
    result = audit(args.bundle); save(args.output, result)
    print({key:result[key] for key in ('status', 'selected_questions',
        'historical_and_test_and_selected_pool', 'cpu_seconds')})
    print(dict(canonical_matches=len(result['canonical_matches']), near_matches=len(result['near_matches'])))


if __name__ == '__main__':
    main()
