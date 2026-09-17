"""Freeze matched-prefix diagnostic; never infer unmeasured CGRS certainty."""
import hashlib
import json
from pathlib import Path
import numpy as np
from review_wsc_native_cpu import Decoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    run = next((ROOT / '.codex_work/wsc_native_long_complete_20260917/results').rglob('engineering_gate.json')).parent
    source = json.loads((run / 'plan.json').read_text())
    result_path = run / 'RC14_wsc_shadow/result.json'
    records = json.loads(result_path.read_text())['records']
    decoder = Decoder('E:/srtp/srtp-final/.codex_work/label_audit_30_20260910/tokenizer.json')
    boundaries = {i for i, s in decoder.vocab.items() if 'ĊĊ' in s}
    params = source['assets']['rebalance_parameters']
    candidates, selected, inputs = [], [], {str(result_path.relative_to(ROOT)): digest(result_path)}
    for rec in records:
        idx = rec['train_index']; ids = rec['token_ids']
        folder = result_path.parent
        cp, hp, npz = folder / f'{idx}_control.npy', folder / f'{idx}_R_history.npy', folder / f'{idx}.npz'
        for path in (cp, hp, npz): inputs[str(path.relative_to(ROOT))] = digest(path)
        control, history = np.load(cp), np.load(hp)
        assert hashlib.sha256(control.tobytes()).hexdigest() == rec['control_trace_sha256']
        assert hashlib.sha256(history.tobytes()).hexdigest() == rec['R_history_sha256']
        with np.load(npz) as z:
            prompt_len = int(z['prompt_tokens'])
            assert z['selected'].tolist() == ids
        rows = []; previous_mean = None
        for p, token in enumerate(ids):
            if token == 151649: break
            if token not in boundaries or p + 1 >= len(control): continue
            c = float(control[p + 1, 1]); coef = float(control[p + 1, 0])
            assert np.isfinite(c) and np.isfinite(coef)
            variance = None if previous_mean is None else (c - previous_mean) ** 2 / 4
            previous_mean = c
            row = dict(train_index=idx, problem_sha256=rec['problem_sha256'],
                boundary_generated_index=p, accepted_generated_tokens=p+1,
                prompt_tokens=prompt_len, prefix_tokens=prompt_len+p+1,
                generated_prefix_sha256=hashlib.sha256(json.dumps(ids[:p+1]).encode()).hexdigest(),
                rebalance_confidence=c, rebalance_high=c > params['q75c'],
                two_step_variance_from_saved_means=variance,
                joint_high_low_variance=None if variance is None else c > params['q75c'] and variance < params['q25v'],
                native_next_coefficient=coef, native_extend=coef > 0,
                clean_opening=bool(control[p+1, 2]), cgrs_certainty=None)
            assert row['prefix_tokens'] <= len(history)
            rows.append(row)
        assert len(rows) >= 16
        candidates.extend(rows)
        ranked = sorted(rows, key=lambda r: hashlib.sha256(f"certainty_overlap_v1:42:{idx}:{r['boundary_generated_index']}".encode()).digest())
        selected.extend(sorted(ranked[:16], key=lambda r:r['boundary_generated_index']))
    assert len(selected) == 128
    primary_keys = {(r['train_index'], r['boundary_generated_index']) for r in selected}
    supplemental = [r for r in candidates if r['rebalance_high'] and (r['train_index'], r['boundary_generated_index']) not in primary_keys]
    for row in selected: row['sample_role'] = 'primary_question_balanced'
    for row in supplemental: row['sample_role'] = 'supplement_all_R_high'
    selected += supplemental
    def counts(rows):
        return dict(steps=len(rows), rebalance_high=sum(r['rebalance_high'] for r in rows),
            native_extend=sum(r['native_extend'] for r in rows),
            joint_high_low_variance=sum(r['joint_high_low_variance'] is True for r in rows))
    plan = dict(run_id='certainty_overlap174_20260917', status='CPU_manifest_only_GPU_not_run',
        purpose='Matched-prefix confidence diagnostic, not efficacy or independent confirmation',
        model=source['assets']['model'], assets=source['assets'], question_count=8, steps=len(selected),
        selection='Primary: 16 boundaries per exposed training question, SHA256 rank fixed seed42, independent of scores. Supplement: every remaining R-high boundary. Report strata separately; enriched union is not a prevalence sample.',
        amendment='CPU-only v1 selected 128 and included one R-high. Before any C probe, add remaining 46 R-high steps to estimate overlap in this fixed R-high population. Preserve original plan.json.',
        thresholds=dict(rebalance_high_strict_gt=params['q75c'], cgrs_high_strict_gt=.9),
        probe=dict(prompt='\n**Final Answer**\n\\boxed', prompt_source='pinned official code; formatting differs from paper prose',
            max_new_tokens=32, temperature=0, seed=42, fresh_rebalance_and_lexical_intervention=False,
            history='Replay saved historical steering for prefix; zero new steering on probe suffix and descendants',
            entropy='Full vocabulary FP32 entropy of raw logits before temperature/top-p; normalize by log(vocab_size)',
            primary_score='Mean entropy of all generated probe tokens through first complete box inclusive; paper Eq1 whole probe response',
            secondary_score='Existing boxed_certainty interior-token score, explicitly an adaptation; never substitute for primary',
            invalid='Missing/empty/unclosed box, nonfinite entropy or alignment failure: report separately, never count as low certainty'),
        report=['2x2 high/not-high counts among valid pairs', 'both-high/all-valid',
            'C-high conditional on R-high and reverse', 'C-high with native positive R coefficient',
            'per-question counts and Spearman correlation', 'invalid counts by R stratum',
            'question-cluster bootstrap descriptive 95% intervals; only eight exposed clusters'],
        gates=['Reconstruct original prompt token count and saved history exactly',
            'Check replay next-token raw distribution against native reference before interpretation; existing BF16 replay failure is not waived',
            'Stop on missing history, preemption, nonfinite logits, or failed replay gate; do not fall back silently to unsteered prefix',
            'If invalid probes exceed 20%, stop interpretation without changing prompt or cap'],
        maximum_probe_generated_tokens=32*len(selected), prefix_prefill_tokens=sum(r['prefix_tokens'] for r in selected),
        expected_gpu_minutes='2-6 after replay gate; estimate only, 10-minute wall stop; no main trajectory regeneration',
        limitation='Already-exposed eight MATH training questions, correlated steps, question-balanced sample not population step prevalence; no correctness or synergy inference',
        source_population=counts(candidates), selected_counts=counts(selected), primary_counts=counts(selected[:128]), rows=selected,
        inputs_sha256=inputs, preparation_source_sha256=digest(Path(__file__)))
    out = HERE / 'certainty_overlap_20260917'
    out.mkdir(exist_ok=True)
    with (out/'plan_v2.json').open('x', encoding='utf8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k:plan[k] for k in ('status','source_population','selected_counts','prefix_prefill_tokens','maximum_probe_generated_tokens')}, indent=2))

if __name__ == '__main__': main()
