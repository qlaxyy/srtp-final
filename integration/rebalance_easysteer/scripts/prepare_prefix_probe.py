"""Freeze bounded short probes of saved calibration prefixes, never new traces."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tarfile

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prefix_probe_policy import checkpoints, test_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), 'Immutable plan already exists')
    prior = read(ROOT/BASE/'configs/seal_comparison130_20260912/plan.json')
    inputs = read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']['inputs']
    with tarfile.open(ROOT/inputs['archive']) as archive:
        raw = archive.extractfile(inputs['member']).read()
    require(hashlib.sha256(raw).hexdigest() == inputs['generations_sha256'], 'Saved500 changed')
    saved = [json.loads(line) for line in raw.splitlines()]
    require(len(saved) == 500, 'Wrong original calibration count')
    tokenizer = read(ROOT/inputs['tokenizer'])
    require(sha(ROOT/inputs['tokenizer']) == prior['model_files_sha256']['tokenizer.json'], 'Tokenizer changed')
    vocab = tokenizer['model']['vocab']
    boundary = {i for text, i in vocab.items() if 'ĊĊ' in text}
    end = next(item['id'] for item in tokenizer['added_tokens'] if item['content'] == '</think>')
    thinking = [r['token_ids'].index(end) if end in r['token_ids'] else len(r['token_ids']) for r in saved]
    eligible = [i for i, length in enumerate(thinking) if length >= 4096]
    chosen = random.Random(20260917).sample(eligible, 32)
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    selected = []
    for index in chosen:
        row = saved[index]
        require(row['problem'] == train[row['train_index']]['problem'], 'Gold pairing mismatch')
        require(len(row['token_ids']) <= 16000, 'Original cap changed')
        points = checkpoints(row['token_ids'], boundary, end)
        require(len(points) <= 15 and all(k <= thinking[index] for k in points), 'Bad causal checkpoints')
        selected.append(dict(calibration_index=index, train_index=row['train_index'],
                             thinking_tokens=thinking[index], total_tokens=len(row['token_ids']),
                             prompt_tokens=len(row['prompt_token_ids']), checkpoints=points))
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n', encoding='utf-8', newline='\n')
    save(out/'selected.json', selected)
    sources = set(prior['source_sha256']) | {BASE+'scripts/'+name for name in
        ['prefix_probe_policy.py', 'prepare_prefix_probe.py', 'run_prefix_probe.py']}
    plan = dict(status='fixed_before_any_trial_answers', purpose='Development-only causal prefix-answer feasibility, not online compression evidence',
        selection_seed=20260917, questions=32, minimum_saved_thinking_tokens=4096,
        eligible_questions=len(eligible), selected_sha256=sha(out/'selected.json'),
        selection='Random32 among original calibration trajectories with >=4096 thinking tokens; no correctness or prefix-answer selection. Length-biased development sample.',
        saved_answers=inputs, remote_saved_answers='/root/autodl-tmp/results/easysteer/own_calibration_500_20260908/generations.jsonl',
        train_sha256=sha(train_path, source=True), model=prior['model'], model_files_sha256=prior['model_files_sha256'],
        source_sha256={name:sha(ROOT/name, source=True) for name in sorted(sources)},
        checkpoint_spacing=1024, consecutive_nonempty_equal_answers=3,
        checkpoint_rule='First generated delimiter at least1024tokens after previous checkpoint, before saved first </think>; exact original token IDs, no future text in each probe.',
        forced_suffix='</think>\nThe answer is \\boxed{',
        trial_max_tokens=128, trial_temperature=0., trial_top_p=1., trial_seed=42,
        number_trial_answers=sum(len(row['checkpoints']) for row in selected),
        maximum_new_trial_tokens=128*sum(len(row['checkpoints']) for row in selected),
        exact_prefix_input_tokens_excluding_suffix=sum(row['prompt_tokens']+k for row in selected for k in row['checkpoints']),
        runtime=dict(max_model_len=32768, max_num_seqs=64, max_num_batched_tokens=32768,
                     gpu_memory_utilization=.9, async_scheduling=True, chunked_prefill=False, prefix_caching=False),
        cpu_policy_checks=test_policy(), expected_gpu_minutes=[3, 10], timeout_seconds=900,
        output='/root/autodl-tmp/results/easysteer/prefix_probe32_20260912',
        advance_gate=dict(minimum_early_stops=10, maximum_correct_to_incorrect=0,
                          minimum_accounted_generated_token_reduction_percent=10),
        grading='Existing author math grader for saved final answer and each complete nonempty forced box; stopping uses strict textual key only, never gold.',
        stop_conditions=['Any source/model/data mismatch or timeout preserves partial outputs and stops; no automatic regeneration.',
                         'A failed fixed feasibility gate stops this candidate without a spacing/consistency search.',
                         'Even a pass permits only planning an independent actual online comparison, not claiming efficacy.'],
        limitations=['Original500 calibration answers are reused unchanged, not regenerated. Only bounded short counterfactual trial answers are new.',
                     'All checkpoints are acquired for mechanism analysis. Batched probe time is not sequential online-policy latency.',
                     'Hypothetical accounting follows the saved unsteered path and charges all probes through stop/fallback. It cannot verify continuation identity or actual dynamic ReBalance KV history.',
                     'A future online method must share the16000 total generation budget including discarded probe tokens and use an appropriate chunking/control comparison.',
                     'No tuning on these32 outcomes and no independent-efficacy claim from calibration data.'])
    save(out/'plan.json', plan)
    print(json.dumps({key:plan[key] for key in ['questions', 'eligible_questions', 'number_trial_answers', 'maximum_new_trial_tokens', 'exact_prefix_input_tokens_excluding_suffix']}))
    print('plan_sha256', sha(out/'plan.json'))


if __name__ == '__main__':
    main()
