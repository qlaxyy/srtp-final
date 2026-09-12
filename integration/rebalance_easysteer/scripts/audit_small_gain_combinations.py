"""Post-hoc CPU description of completed screens; never changes their gates."""
import argparse
from pathlib import Path
import numpy as np
from prepare_bcc import read, save, sha, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Output exists')
    base = a.root/'.codex_work/mechanism_candidates_20260911'
    folder = base/'gpu_run_20260912'
    analysis = read(folder/'analysis.json')
    for name, digest in analysis['source_sha256'].items():
        require(sha(folder/name) == digest, 'Changed source: '+name)
    groups = {}
    for name in ['original_dynamic', 'min_displacement', 'orthogonal_mean']:
        raw = read(folder/(name+'.json'))['rebalance_dynamic']
        grade = read(folder/(name+'.author.json'))
        require(grade['input_sha256'] == sha(folder/(name+'.json')), 'Grade linkage')
        rows = raw['records']
        require(len(rows) == len(grade['records']) == 100, 'Scope')
        for r in rows:
            ids = r['token_ids']
            require(len(ids) == r['tokens'] <= 16000, 'Total tokens')
            require(r['thinking_tokens'] == (ids.index(151649) if 151649 in ids else len(ids)), 'Thinking tokens')
        groups[name] = dict(rows=rows, grades=grade['records'],
            total=np.array([r['tokens'] for r in rows]),
            thinking=np.array([r['thinking_tokens'] for r in rows]),
            correct=np.array([int(r['correct']) for r in grade['records']]))
    sample = np.random.default_rng(20260912).integers(0, 100, (10000, 100))
    x = groups['original_dynamic']; results = {}
    for name in ['min_displacement', 'orthogonal_mean']:
        y = groups[name]
        require([r['problem'] for r in x['rows']] == [r['problem'] for r in y['rows']], 'Problem alignment')
        require([r['train_index'] for r in x['grades']] == [r['train_index'] for r in y['grades']], 'Grade alignment')
        metrics = {}
        for metric in ['total', 'thinking']:
            bootstrap = 100*(y[metric][sample].mean(1)/x[metric][sample].mean(1)-1)
            metrics[metric] = dict(percent_change=float(100*(y[metric].mean()/x[metric].mean()-1)),
                paired_bootstrap_95_percent_interval=np.quantile(bootstrap, [.025, .975]).tolist())
        difference = y['correct']-x['correct']
        metrics['accuracy'] = dict(difference_pp=float(100*difference.mean()),
            paired_bootstrap_95_pp_interval=np.quantile(100*difference[sample].mean(1), [.025, .975]).tolist())
        results[name] = metrics
    # Existing minimum-displacement vector is parallel to the old LDA w.
    # P_w(v)=w*(w^T v)/(w^T w). Norm-matching P_w(v) erases all of v's
    # orthogonal information: only the sign remains, whatever BCC learns.
    asset = base/'cpu_completed/min_displacement/direction.npy'
    w = np.load(asset).astype(float).reshape(-1); u = w/np.linalg.norm(w)
    rng = np.random.default_rng(20260912)
    errors = []
    for v in rng.normal(size=(32, len(w))):
        scalar = float(u@v)
        projected = u*scalar
        errors.append(float(np.linalg.norm(projected/np.linalg.norm(projected)-np.sign(scalar)*u)))
    report = dict(status='completed_posthoc_CPU_only', new_answers=0, GPU_calls=0,
        analysis_sha256=sha(folder/'analysis.json'), source_sha256=analysis['source_sha256'],
        vector_sha256=sha(asset), script_sha256=sha(Path(__file__)), seed=20260912,
        bootstrap_repetitions=10000, comparisons=results,
        old_w_projection_collapse_max_error=max(errors),
        combination_decision='Do not combine BCC with the old-w minimum-displacement projection: at matched norm it collapses to the old axis up to sign. A BCC-specific discriminator would be a separate calibrated method.',
        alternatives='Fixed minimum-displacement alone deserves independent confirmation under a new prospective small-gain objective. Norm restoration is mechanistically distinct but has cap regressions; vector averaging has unchosen weights and no additive-gain guarantee.',
        limitations=['Existing selected screens, not new independent evidence.',
                    'Question bootstrap conditions on one generation seed; no multiplicity correction.',
                    'Old 5 percent gates and stopped decisions remain unchanged.',
                    'No successful BCC vector exists yet; synthetic algebra is not an efficacy test.'])
    save(a.output, report)
    print(results)


if __name__ == '__main__':
    main()
