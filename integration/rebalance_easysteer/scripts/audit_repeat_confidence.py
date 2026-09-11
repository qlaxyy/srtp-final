"""Describe reviewed step content versus saved confidence; no model execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'integration/rebalance_easysteer/configs'
PLAN = {
    'date': '2026-09-11',
    'purpose': 'Check whether reviewed repetition shares high-confidence/low-variance signals with new reasoning, using saved current-step statistics.',
    'sample': 'All existing 30 training calibration questions and 76 reviewed checkpoints; no new selection or labels. All seven capped answers remain included.',
    'timing': 'Use the reviewed step itself (outgoing statistics), not its incoming boundary. End-of-thinking steps still have descriptive statistics but no injection opportunity. First-step variance is defined as zero and reported separately.',
    'regions': 'Strict confidence > frozen q75c and variance < frozen q25v. Other regions use the complementary inequalities. Positive coefficients use the existing 1e-6 reporting tolerance. No fitted thresholds.',
    'groups': 'Report new-supported, new-other, repeat, uncertain separately; also new-all. Split by original label and finish reason. Question-balanced fractions average within-question fractions, not a population estimate.',
    'within_question': 'Use every question containing both reviewed repeat and supported-new checkpoints. Compare mean confidence, high/low-region fractions and eligible-positive fractions; no significance or population claim.',
    'exact_history': 'For each reviewed repeat checkpoint, find all earlier complete steps with exactly equal evidence_text after outer whitespace stripping. Preserve internal whitespace, case, numbers and operators. Compare consecutive occurrence confidences with a 1e-12 numerical tie tolerance, and first-to-target change. Same wording does not guarantee the same semantic role; histories end at the reviewed checkpoint.',
    'limitations': [
        'Original sampling was stratified by confidence/lexicon labels. Counts cannot estimate prevalence or classifier precision.',
        'Content labels are existing single-reviewer evidence, not semantic gold or validated over/underthinking states.',
        'Saved traces are unsteered. Projected positive coefficients are not observed interventions or causal effects.',
        'High-confidence/low-variance is a descriptive region, not a hard branch of the smooth controller.',
        'No new generation, replay of model hidden states, GPU, label replacement, threshold search or frozen-result change.',
    ],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def summarize(rows):
    by_question = {q: [r for r in rows if r['question'] == q]
                   for q in sorted({r['question'] for r in rows})}
    eligible = [r for r in rows if r['has_boundary']]
    return {
        'steps': len(rows), 'questions': len(by_question),
        'high_c_low_v_steps': sum(r['high_c_low_v'] for r in rows),
        'regions': dict(sorted(Counter(r['region'] for r in rows).items())),
        'confidence_median': statistics.median(r['confidence'] for r in rows) if rows else None,
        'variance_median': statistics.median(r['variance'] for r in rows) if rows else None,
        'first_steps_with_conventional_zero_variance': sum(r['step'] == 1 for r in rows),
        'eligible_boundaries': len(eligible),
        'positive_eligible_boundaries': sum(r['positive'] for r in eligible),
        'high_c_low_v_eligible_boundaries': sum(r['high_c_low_v'] for r in eligible),
        'positive_eligible_outside_strict_high_low': sum(r['positive'] and not r['high_c_low_v'] for r in eligible),
        'question_balanced_high_low_fraction': statistics.mean(
            sum(r['high_c_low_v'] for r in group)/len(group)
            for group in by_question.values()) if rows else None,
        'question_counts': {q: {'steps': len(group),
            'high_c_low_v_steps': sum(r['high_c_low_v'] for r in group),
            'positive_eligible_boundaries': sum(r['has_boundary'] and r['positive'] for r in group)}
            for q, group in by_question.items()},
    }


def run(output, report_path):
    start = time.perf_counter()
    if output.exists() or report_path.exists():
        raise FileExistsError('Use new output and report paths; old artifacts are immutable')
    output.mkdir(parents=True)
    save_new(output / 'plan.json', PLAN)
    alignment_path = CONFIG / 'calibration_control_alignment30_20260910.json'
    evidence_path = CONFIG / 'calibration_label_evidence30_20260910.json'
    fit_path = CONFIG / 'auto_code_v2_1p5b_20260908.json'
    frozen_path = CONFIG / 'final_results_20260909.json'
    alignment, evidence, fit, frozen = map(read, [alignment_path, evidence_path, fit_path, frozen_path])
    paths = {'alignment': alignment_path, 'evidence': evidence_path, 'fit': fit_path,
             'frozen_manifest': frozen_path,
             'projected_steps': ROOT / alignment['artifacts']['projected_steps_path'],
             'evidence_index': ROOT / evidence['artifacts']['index_path']}
    for key, expected in [('evidence', alignment['artifacts']['evidence_sha256']),
                          ('fit', alignment['artifacts']['fit_sha256']),
                          ('frozen_manifest', alignment['artifacts']['frozen_manifest_sha256']),
                          ('projected_steps', alignment['artifacts']['projected_steps_sha256']),
                          ('evidence_index', evidence['artifacts']['index_sha256'])]:
        assert sha(paths[key]) == expected, key
    runtime_path = 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
    blob = subprocess.check_output(['git', 'show', 'HEAD:' + runtime_path], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == frozen['source_sha256'][runtime_path]
    projected = {q['question']: q for q in read(paths['projected_steps'])}
    indexed = {q['question']: q for q in read(paths['evidence_index'])['questions']}
    evidence_points = {p['case_id']: p for q in evidence['questions'] for p in q['checkpoints']}
    p = fit['parameters']
    assert p == alignment['parameters'] and len(projected) == len(indexed) == 30
    rows, histories = [], []
    for checkpoint in alignment['checkpoints']:
        qid, number = checkpoint['question'], checkpoint['step']
        q, ix = projected[qid], indexed[qid]
        step, text = q['steps'][number-1], ix['steps'][number-1]
        review = evidence_points[checkpoint['case_id']]
        assert review['progress_evidence'] == checkpoint['progress_evidence']
        assert review['math_validity'] == checkpoint['math_validity']
        assert q['train_index'] == ix['train_index'] == checkpoint['train_index']
        assert step['step'] == text['step'] == number
        assert step['start'] == text['start'] == review['start']
        assert step['stop'] == text['stop'] == review['stop']
        if step['has_boundary']:
            for key in ['confidence', 'variance', 'coefficient']:
                assert step[key] == checkpoint['outgoing'][key]
        c, v = step['confidence'], step['variance']
        expected_v = 0. if number == 1 else (c-q['steps'][number-2]['confidence'])**2/4
        assert abs(v-expected_v) < 1e-15
        high, low = c > p['q75c'], v < p['q25v']
        category = checkpoint['progress_evidence']
        if category == 'new':
            category = 'new_supported' if checkpoint['math_validity'] == 'supported' else 'new_other'
        row = {key: checkpoint[key] for key in ['question', 'train_index', 'case_id', 'step',
            'progress_evidence', 'math_validity', 'original_label', 'original_label_route']}
        row.update(category=category, finish_reason=q['finish_reason'], confidence=c, variance=v,
            coefficient=step['coefficient'], has_boundary=step['has_boundary'],
            positive=step['coefficient'] > alignment['sign_zero_tolerance'], high_c_low_v=high and low,
            region=('high_c' if high else 'not_high_c') + ('_low_v' if low else '_not_low_v'),
            text=text['evidence_text'], prior_steps=review['prior_steps'],
            prior_evidence=review['prior_evidence'], progress_reason=review['progress_reason'])
        rows.append(row)
        if category == 'repeat':
            matches = [s for s in ix['steps'][:number] if s['evidence_text'].strip() == text['evidence_text'].strip()]
            if len(matches) > 1:
                occurrences = [{key: q['steps'][s['step']-1][key] for key in
                    ['step', 'confidence', 'variance', 'coefficient', 'has_boundary']} for s in matches]
                differences = [b['confidence']-a['confidence'] for a,b in zip(occurrences, occurrences[1:])]
                histories.append(dict(case_id=row['case_id'], question=qid, target_step=number,
                    text=text['evidence_text'], occurrences=occurrences,
                    increases=sum(d > 1e-12 for d in differences), decreases=sum(d < -1e-12 for d in differences),
                    numerical_ties=sum(abs(d) <= 1e-12 for d in differences),
                    first_to_target_confidence_change=occurrences[-1]['confidence']-occurrences[0]['confidence']))
    assert len(rows) == 76 and len({r['case_id'] for r in rows}) == 76
    categories = ['repeat', 'new_supported', 'new_other', 'uncertain', 'new_all']
    def select(category):
        return [r for r in rows if r['category'] == category or category == 'new_all' and r['progress_evidence'] == 'new']
    groups = {category: summarize(select(category)) for category in categories}
    matched = []
    for qid in sorted(projected):
        repeats = [r for r in select('repeat') if r['question'] == qid]
        new = [r for r in select('new_supported') if r['question'] == qid]
        if repeats and new:
            matched.append(dict(question=qid, train_index=projected[qid]['train_index'],
                repeat=summarize(repeats), new_supported=summarize(new),
                mean_confidence_difference_repeat_minus_new=statistics.mean(r['confidence'] for r in repeats)-statistics.mean(r['confidence'] for r in new)))
    result = dict(status='completed_cpu_descriptive_association_not_intervention_test',
        plan=PLAN, execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        question_count=30, reviewed_steps=76, capped_questions=sum(q['finish_reason']=='length' for q in projected.values()),
        new_generations=0, model_forwards=0, gpu_tasks=0, thresholds=dict(q75c=p['q75c'], q25v=p['q25v'],
            coefficient_zero_tolerance=alignment['sign_zero_tolerance']),
        groups=groups, by_original_label={label: {category: summarize([r for r in select(category) if r['original_label']==label]) for category in categories}
            for label in sorted({r['original_label'] for r in rows})},
        by_finish_reason={finish: {category: summarize([r for r in select(category) if r['finish_reason']==finish]) for category in categories}
            for finish in sorted({r['finish_reason'] for r in rows})},
        matched_questions=matched, exact_text_histories=histories, records=rows,
        provenance={name: dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)) for name,path in paths.items()},
        script_sha256=sha(__file__), elapsed_seconds=time.perf_counter()-start)
    save_new(output/'summary.json', result)
    save_new(report_path, result)
    print(json.dumps({k: result[k] for k in ['question_count','reviewed_steps','capped_questions','thresholds','groups','elapsed_seconds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.codex_work/repeat_confidence30_20260911')
    parser.add_argument('--report', type=Path, default=CONFIG/'repeat_confidence30_20260911.json')
    args = parser.parse_args()
    run(args.output, args.report)
