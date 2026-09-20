"""Describe saved R/RS trajectories without model replay or parameter selection."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert read(args.run/'batch_status.json')['status'] == 'complete'
    resolved = read(args.run/'resolved_plan.json')
    results = {}
    for role in resolved['expansion']['roles']:
        paths = [args.run/role/arm/'result.json' for arm in ('R', 'RS')]
        groups = [read(p)['records'] for p in paths]
        rows = []
        for base, combined in zip(*groups):
            assert base['train_index'] == combined['train_index']
            a, b = base['token_ids'], combined['token_ids']
            h = combined['hybrid']
            first = h['first_bias']
            end = a.index(151649) if 151649 in a else -1
            common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                          min(len(a), len(b)))
            prefix_length = first if first >= 0 else min(len(a), len(b))
            rows.append(dict(
                train_index=base['train_index'], token_identical=a == b,
                R_history_identical=base['R_history']['sha256'] == combined['R_history']['sha256'],
                common_prefix_tokens=common,
                first_difference=(common if a != b else -1),
                prefix_before_actual_bias_identical=a[:prefix_length] == b[:prefix_length],
                R_natural_end=end, RS_end=h['end_position'], first_bias=first,
                first_bias_before_R_end=first >= 0 and (end < 0 or first < end),
                first_bias_at_R_end=first >= 0 and first == end,
                worker_bias_count=h['bias_count'],
                worker_revived_end_count=h['revived_end_count'],
                worker_trigger_count=h['trigger_count'],
                token_delta=len(b)-len(a),
                thinking_delta=combined['thinking_tokens']-base['thinking_tokens']))
        results[role] = dict(
            n=len(rows),
            token_identical=sum(x['token_identical'] for x in rows),
            R_history_identical=sum(x['R_history_identical'] for x in rows),
            changed_ids=[x['train_index'] for x in rows if not x['token_identical']],
            shorter=sum(x['token_delta'] < 0 for x in rows),
            longer=sum(x['token_delta'] > 0 for x in rows),
            unchanged_length=sum(x['token_delta'] == 0 for x in rows),
            first_bias_before_R_end=sum(x['first_bias_before_R_end'] for x in rows),
            first_bias_at_R_end=sum(x['first_bias_at_R_end'] for x in rows),
            formal_prefix_mismatches=[x['train_index'] for x in rows
                                      if not x['prefix_before_actual_bias_identical']],
            source_sha256={str(p.relative_to(args.run)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in paths},
            records=rows)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(datasets=results, limitations=[
            'Saved outputs only; no per-step logits or new model forward.',
            'Worker intervention counts may include one discarded asynchronous tail token.',
            'A revived end token or an earlier end is not evidence of answer correctness.'
        ]), stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
