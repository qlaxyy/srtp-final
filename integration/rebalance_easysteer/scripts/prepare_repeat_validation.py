"""Freeze one fresh training validation set on CPU; never load a model."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = 'integration/rebalance_easysteer/'
TRAIN = 'sources/ReBalance/Data/Math_Train/test.jsonl'
TESTS = ('Math_Math500', 'Math_GSM8K')
SEED = 20260911
COUNT = 100


def sha(path, source=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if source else raw).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def norm(text):
    return re.sub(r'\s+', '', text)


def select(train, held_out, calibration, previous):
    excluded = set(calibration) | set(previous)
    forbidden = {norm(r['problem']) for r in held_out}
    forbidden.update(norm(train[i]['problem']) for i in excluded)
    eligible = []
    seen = set(forbidden)
    for i, row in enumerate(train):
        text = norm(row['problem'])
        if i not in excluded and text not in seen:
            eligible.append(i)
            seen.add(text)
    return random.Random(SEED).sample(eligible, COUNT), len(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prior = json.loads(args.prior_manifest.read_text(encoding='utf-8'))
    old = json.loads((ROOT / BASE / 'configs/first_step_validation100_20260910.json').read_text())
    if sha(args.prior_manifest) != old['hashes']['selection_manifest_sha256']:
        raise ValueError('Previous validation manifest changed')
    train = rows(ROOT / TRAIN)
    if sha(ROOT / TRAIN, source=True) != prior['train_sha256'] or len(train) != 7500:
        raise ValueError('Training source changed')
    tests = {name: rows(ROOT / f'sources/ReBalance/Data/{name}/test.jsonl') for name in TESTS}
    # Reconstruct the original selection, not the answers or calibration fit.
    forbidden_math = {norm(r['problem']) for r in tests['Math_Math500']}
    eligible_cal = [i for i, r in enumerate(train) if norm(r['problem']) not in forbidden_math]
    calibration = random.Random(42).sample(eligible_cal, 500)
    if sorted(calibration) != sorted(prior['excluded_calibration_indices']):
        raise ValueError('Original calibration selection mismatch')
    previous = old['validation_train_indices']
    if previous != prior['train_indices'] or len(set(previous)) != 100:
        raise ValueError('Previous validation selection mismatch')
    selected, eligible_count = select(
        train, [r for group in tests.values() for r in group], calibration, previous)
    selected_norm = {norm(train[i]['problem']) for i in selected}
    overlap = {name: len(selected_norm & {norm(r['problem']) for r in group})
               for name, group in tests.items()}
    overlap.update(calibration=len(set(selected) & set(calibration)),
                   previous_validation=len(set(selected) & set(previous)))
    if any(overlap.values()) or len(selected_norm) != COUNT:
        raise ValueError('Fresh validation is not independent')
    freeze = json.loads((ROOT / BASE / 'configs/final_results_20260909.json').read_text())
    code = dict(freeze['source_sha256'])
    for rel in ('eval/rebalance_dynamic_eval.py', 'eval/repeat_positive_gate.py',
                'scripts/prepare_repeat_validation.py', 'scripts/run_repeat_validation.py'):
        name = BASE + rel
        code[name] = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    config = json.loads((ROOT / BASE / 'configs/auto_code_v2_1p5b_20260908.json').read_text())
    args.output.mkdir(parents=True)
    (args.output/'.gitattributes').write_text('* text eol=lf\n', encoding='utf-8', newline='\n')
    dataset = args.output / 'validation100.jsonl'
    dataset.write_text(''.join(json.dumps(dict(train[i], train_index=i), ensure_ascii=False)
                               + '\n' for i in selected), encoding='utf-8', newline='\n')
    plan = dict(
        status='prepared_not_run', scope='Fresh 1.5B MATH training validation; not MATH-500',
        count=COUNT, new_answers_planned=2*COUNT, selection_seed=SEED,
        selection='Uniform sample without replacement after index and whitespace-normalized prompt exclusions; no generated outcomes read',
        train_indices=selected, excluded_calibration_indices=sorted(calibration),
        excluded_previous_validation_indices=previous, eligible_count=eligible_count,
        overlap_checks=overlap, unique_prompts=len(selected_norm),
        train_sha256=sha(ROOT/TRAIN, source=True), dataset_sha256=sha(dataset),
        prior_manifest_sha256=sha(args.prior_manifest),
        test_prompt_source_sha256={name:sha(ROOT/f'sources/ReBalance/Data/{name}/test.jsonl', source=True) for name in TESTS},
        prompt_source_hash_format='SHA256 after CRLF to LF normalization; selected dataset hash is raw bytes',
        vector_sha256=old['hashes']['vector_sha256'], fit_sha256=old['hashes']['fit_sha256'],
        model_files_sha256=old['model_verified_against_freeze'],
        dynamic_parameters=config['parameters'], decoder_output_layer=20,
        model='/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B',
        assets='/root/autodl-tmp/results/easysteer/auto_code_v2_500_20260908',
        runtime=dict(max_tokens=16000, max_model_len=32768, max_num_seqs=128,
                     max_num_batched_tokens=32768, gpu_memory_utilization=0.90,
                     temperature=0.7, top_p=0.95, seed=42, async_scheduling=True,
                     chunked_prefill=False, group_timeout_seconds=0),
        arms={'original_dynamic':'off', 'repeat_cancel_positive':'cancel_positive'},
        run_order=['original_dynamic','repeat_cancel_positive'],
        comparison='Same existing vector/layer/controller, no first-prompt injection; candidate masks only original positive applied scales after complete-step repetition',
        timing='Separate fresh processes, model loading excluded and reported; original has no observer. Candidate CPU synchronization and checkpoint overhead are included. One fixed order, not a repeated speed benchmark.',
        assessment=['Keep all 100 paired answers including incorrect and capped answers.',
                    'Report author accuracy, thinking and total tokens, cap count, pure generation seconds, flips, gate events and preemption/restore counts.',
                    'If no cancellation occurs, report coverage only. If total tokens do not decrease or correct count decreases, do not expand or retune this set.',
                    'Fewer tokens and no observed accuracy decrease justify discussion of a separate held-out evaluation, not a claim of proven accuracy preservation.',
                    'A failed/incomplete arm stays incomplete; preserve raw and partial files, do not regenerate completed arms automatically.'],
        source_sha256=code, source_hash_format='SHA256 after CRLF to LF normalization',
        gpu_started=False, new_generations=0,
        limitations=['CPU hook contracts passed with a state double, not CUDA acceptance.',
                     'Exact text repetition is conservative evidence, not a semantic or correctness label.',
                     'The original 500 calibration answers are development data; this fresh 100 is validation, not formal test data.'])
    (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:plan[k] for k in ('status','count','eligible_count','overlap_checks','dataset_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
