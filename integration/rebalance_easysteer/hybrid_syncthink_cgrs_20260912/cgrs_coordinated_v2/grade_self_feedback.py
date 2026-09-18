"""Grade training feedback; never report training selection as efficacy."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    complete = read(a.run / 'complete.json')
    assert complete['phase'] in ('full', 'collection_merged') and complete['passed']
    if complete['phase'] == 'collection_merged':
        for source in read(a.run/'merge_provenance.json')['sources']:
            assert sha(Path(source['path'])) == source['sha256']
    release = read(a.artifact / 'release.json')
    assert complete['release_sha256'] == sha(a.artifact / 'release.json')
    for name, digest in release['artifact_sha256'].items():
        assert sha(a.artifact / name) == digest, name
    assert sha(a.run / 'plan.json') == release['artifact_sha256']['plan.json']
    assert sha(a.run / 'release.json') == sha(a.artifact / 'release.json')
    plan = read(a.artifact / 'plan.json')
    assert plan['experiment_kind'] == 'self_feedback_collection_v1'
    graders = {'parser.py': '8d04ceb93e9ef50d61157fd1c8826a5cbf8d7f64cda85459f1166e19d8784208',
               'grader.py': '0338f6549f2cb98806473e746baf40b00d9c5bd21a69bc7de0008e4ad4c8d768'}
    for name, digest in graders.items():
        assert sha(a.runtime_root / 'sources/ReBalance/utils' / name) == digest
    sys.path.insert(0, str(a.runtime_root / 'sources/ReBalance'))
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    data = read(a.run / 'L27_L27/result.json')
    assert data['status'] == 'complete'
    groups = {'U': read(a.artifact / 'baseline.json'), 'L27': data['records']}
    assert all(len(x) == len(plan['rows']) == 500 for x in groups.values())
    labels = [dict(dataset_index=i, problem_sha256=r['problem_sha256']) for i, r in enumerate(plan['rows'])]
    for name, records in groups.items():
        path = a.run / (name + '_author_partial.jsonl')
        previous = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if a.resume and path.exists() else []
        assert len(previous) <= 500
        with path.open('a' if a.resume else 'x', encoding='utf8') as f:
            for i, (row, r) in enumerate(zip(plan['rows'], records)):
                assert r['dataset_index'] == i and r['problem_sha256'] == row['problem_sha256']
                ids = r['token_ids']
                assert 0 < len(ids) <= 16000
                identity = dict(dataset_index=i, problem_sha256=row['problem_sha256'], text_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
                if i < len(previous):
                    label = previous[i]
                    assert all(label[k] == v for k, v in identity.items())
                else:
                    _, gold = parse_ground_truth(row, 'math')
                    answer = extract_answer(r['text'])
                    # Exceptions interrupt grading; they are not scored as incorrect.
                    label = dict(identity, correct=bool(check_is_correct(answer, gold)))
                    f.write(json.dumps(label) + '\n')
                    f.flush()
                labels[i][name] = dict(label, tokens=len(ids), thinking_tokens=ids.index(151649) if 151649 in ids else len(ids),
                                       closed=151649 in ids and r['finish_reason'] != 'length', finish_reason=r['finish_reason'])
    summaries = {}
    for name in groups:
        rs = [r[name] for r in labels]
        summaries[name] = dict(count=500, correct=sum(r['correct'] for r in rs),
                              mean_tokens=sum(r['tokens'] for r in rs)/500,
                              mean_thinking_tokens=sum(r['thinking_tokens'] for r in rs)/500,
                              capped=sum(r['finish_reason']=='length' or r['tokens']==16000 for r in rs))
    save(a.run/'labels.json', labels)
    save(a.run/'training_analysis.json', dict(status='Training feedback only; no candidate efficacy evaluation', groups=summaries,
         right_to_wrong=[r['dataset_index'] for r in labels if r['U']['correct'] and not r['L27']['correct']],
         wrong_to_right=[r['dataset_index'] for r in labels if not r['U']['correct'] and r['L27']['correct']],
         generation_seconds=data['generation_seconds'], timing_note=data.get('timing_note'), extra_model_forward_count=data['extra_model_forward_count'],
         grading_seconds=time.monotonic()-started, grading_timing_note='This invocation only; excludes any prior cached grading when --resume is used.', grader_sha256=graders, analysis_script_sha256=sha(Path(__file__)),
         release_sha256=sha(a.artifact/'release.json'), baseline_sha256=sha(a.artifact/'baseline.json'),
         candidate_result_sha256=sha(a.run/'L27_L27/result.json'), labels_sha256=sha(a.run/'labels.json')))
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
