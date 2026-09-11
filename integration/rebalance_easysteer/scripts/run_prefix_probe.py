"""Acquire short causal trial answers and separately grade their feasibility."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prefix_probe_policy import checkpoints, boxed_key, account, test_policy


def validate(bundle):
    plan = read(bundle/'plan.json')
    selected = read(bundle/'selected.json')
    require(plan['status'] == 'fixed_before_any_trial_answers', 'Wrong protocol')
    require(len(selected) == plan['questions'] == 32, 'Scope changed')
    require(sha(bundle/'selected.json') == plan['selected_sha256'], 'Selection changed')
    require(plan['trial_max_tokens'] == 128 and plan['checkpoint_spacing'] == 1024,
            'Unplanned probe search')
    require(plan['consecutive_nonempty_equal_answers'] == 3, 'Unplanned threshold')
    require(sum(len(r['checkpoints']) for r in selected) == plan['number_trial_answers'],
            'Checkpoint count changed')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT/name, source=True) == digest, 'Source changed: '+name)
    test_policy()
    return plan, selected


def saved_inputs(plan):
    path = Path(plan['remote_saved_answers'])
    require(sha(path) == plan['saved_answers']['generations_sha256'], 'Saved500 changed')
    saved = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    require(sha(train_path, source=True) == plan['train_sha256'], 'Training input changed')
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    require(len(saved) == 500, 'Missing original answers')
    return saved, train


def grade(plan, selected, bundle, out):
    ledger = read(out/'ledger.json')
    require(ledger['status'] == 'probes_completed_grading_pending', 'Incomplete probes')
    require(ledger['plan_sha256'] == sha(bundle/'plan.json'), 'Plan mismatch')
    require(not (out/'analysis.json').exists(), 'Grading already complete')
    require(sha(out/'probes.json') == ledger['probes_sha256'], 'Trial answers changed')
    saved, train = saved_inputs(plan)
    probes = read(out/'probes.json')['records']
    sys.path.insert(0, str(ROOT/'sources/ReBalance'))
    from utils.parser import extract_answer, parse_ground_truth
    from utils.grader import check_is_correct
    by_question = {row['calibration_index']: [] for row in selected}
    for probe in probes:
        require(boxed_key(probe['text']) == probe['key'], 'Answer key changed')
        by_question[probe['calibration_index']].append(probe)
    analyses = []
    started = time.perf_counter()
    with (out/'author.partial.jsonl').open('x', encoding='utf-8') as stream:
        for item in selected:
            index = item['calibration_index']
            original = saved[index]
            require(original['train_index'] == item['train_index'], 'Question mismatch')
            source = train[item['train_index']]
            require(source['problem'] == original['problem'], 'Gold mismatch')
            _, gold = parse_ground_truth(source, 'math')
            original_correct = bool(check_is_correct(extract_answer(original['text'], 'math'), gold))
            trial = by_question[index]
            require([p['prefix_tokens'] for p in trial] == item['checkpoints'], 'Order changed')
            trial_correct = [bool(p['key']) and bool(check_is_correct(p['key'], gold)) for p in trial]
            costs = account(item['total_tokens'], item['thinking_tokens'], item['checkpoints'],
                            [p['tokens'] for p in trial], [p['key'] for p in trial],
                            [p['prompt_tokens'] for p in trial])
            stop = costs['stop_index']
            selected_correct = original_correct if stop is None else trial_correct[stop]
            row = dict(**item, original_correct=original_correct,
                       hypothetical_policy_correct=selected_correct,
                       trial_correct=trial_correct,
                       trial_keys=[p['key'] for p in trial], accounting=costs)
            analyses.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
    originals = sum(r['total_tokens'] for r in selected)
    accounted = sum(r['accounting']['hypothetical_generated_tokens'] for r in analyses)
    early = [r['calibration_index'] for r in analyses if r['accounting']['stop_index'] is not None
             and r['accounting']['hypothetical_thinking_tokens'] < r['thinking_tokens']]
    degraded = [r['calibration_index'] for r in analyses if r['original_correct'] and not r['hypothetical_policy_correct']]
    improved = [r['calibration_index'] for r in analyses if not r['original_correct'] and r['hypothetical_policy_correct']]
    reduction = 100*(1-accounted/originals)
    gate = plan['advance_gate']
    passes = (len(early) >= gate['minimum_early_stops']
              and len(degraded) <= gate['maximum_correct_to_incorrect']
              and reduction >= gate['minimum_accounted_generated_token_reduction_percent'])
    summary = dict(status='completed_development_feasibility_only',
        plan_sha256=sha(bundle/'plan.json'), questions=32, trial_answers=len(probes),
        original_correct=sum(r['original_correct'] for r in analyses),
        hypothetical_policy_correct=sum(r['hypothetical_policy_correct'] for r in analyses),
        original_mean_thinking_tokens=sum(r['thinking_tokens'] for r in selected)/32,
        original_mean_total_tokens=originals/32,
        hypothetical_mean_thinking_tokens=sum(r['accounting']['hypothetical_thinking_tokens'] for r in analyses)/32,
        hypothetical_mean_accounted_generated_tokens=accounted/32,
        hypothetical_generated_token_reduction_percent=reduction,
        early_stop_indices=early, improved_indices=improved, degraded_indices=degraded,
        original_capped=sum(saved[r['calibration_index']]['finish_reason'] == 'length' or r['total_tokens'] == 16000 for r in selected),
        trial_capped=sum(p['finish_reason'] == 'length' or p['tokens'] == 128 for p in probes),
        incomplete_or_empty_box_count=sum(not p['key'] for p in probes),
        total_acquired_probe_tokens=sum(p['tokens'] for p in probes),
        total_acquired_prefill_tokens=sum(p['prompt_tokens'] for p in probes),
        hypothetical_used_probe_tokens=sum(r['accounting']['probe_output_tokens'] for r in analyses),
        hypothetical_used_probe_prefill_tokens=sum(r['accounting']['probe_prefill_input_tokens'] for r in analyses),
        passes_fixed_feasibility_gate=passes,
        decision='Plan independent actual online test; no efficacy claim' if passes else 'Stop candidate without changing spacing or agreement count',
        generation_seconds=ledger['generation_seconds'], grading_seconds=time.perf_counter()-started,
        grader_sha256={name:sha(ROOT/'sources/ReBalance/utils'/name, source=True) for name in ['parser.py', 'grader.py']},
        limitations=plan['limitations'], per_question=analyses)
    save(out/'analysis.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_question', 'limitations']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--grade-only', action='store_true')
    args = parser.parse_args()
    bundle, out = args.bundle.resolve(), args.output.resolve()
    plan, selected = validate(bundle)
    if args.grade_only:
        grade(plan, selected, bundle, out)
        return
    if not args.execute:
        print(json.dumps(dict(status='cpu_preflight_passed', questions=32,
                              trials=plan['number_trial_answers'], model_calls=0)))
        return
    require(sys.platform == 'linux' and not out.exists(), 'Fresh Linux output required')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Dirty code')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Other GPU work')
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', '')
    saved, _ = saved_inputs(plan)
    for name, digest in plan['model_files_sha256'].items():
        require(sha(Path(plan['model'])/name) == digest, 'Model asset changed')
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    require(torch.cuda.is_available(), 'GPU unavailable; no reinstall')
    require(Path(vllm.__file__).resolve() == (ROOT/'sources/EasySteer/vllm-steer/vllm/__init__.py').resolve(), 'Wrong runtime')
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    boundary = {i for text, i in tokenizer.get_vocab().items() if 'ĊĊ' in text}
    end_ids = tokenizer.encode('</think>', add_special_tokens=False)
    require(len(end_ids) == 1, 'Bad end marker')
    suffix = tokenizer.encode(plan['forced_suffix'], add_special_tokens=False)
    require(suffix[0] == end_ids[0] and tokenizer.decode(suffix) == plan['forced_suffix'], 'Bad forced suffix')
    prompts, mappings = [], []
    for item in selected:
        row = saved[item['calibration_index']]
        require(checkpoints(row['token_ids'], boundary, end_ids[0]) == item['checkpoints'], 'Runtime checkpoint mismatch')
        for prefix in item['checkpoints']:
            ids = row['prompt_token_ids']+row['token_ids'][:prefix]+suffix
            require(len(ids)+128 <= 32768 and end_ids[0] not in row['token_ids'][:prefix], 'Noncausal prompt')
            prompts.append(dict(prompt_token_ids=ids))
            mappings.append(dict(calibration_index=item['calibration_index'], train_index=item['train_index'], prefix_tokens=prefix))
    require(len(prompts) == plan['number_trial_answers'], 'Missing probes')
    out.mkdir(parents=True)
    ledger = dict(status='loading_model', started_unix=time.time(), plan_sha256=sha(bundle/'plan.json'),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  saved_original_answers_sha256=plan['saved_answers']['generations_sha256'],
                  complete_calibration_answers_regenerated=0, suffix_token_ids=suffix,
                  torch=torch.__version__, vllm=vllm.__version__, files_sha256={})
    save(out/'ledger.json', ledger)
    watchdog = threading.Timer(plan['timeout_seconds'], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        started = time.perf_counter()
        llm = LLM(model=plan['model'], dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=32768, max_num_seqs=64, max_num_batched_tokens=32768,
                  gpu_memory_utilization=.9, enable_steer_vector=False,
                  enable_chunked_prefill=False, enable_prefix_caching=False,
                  async_scheduling=True, seed=42)
        ledger['startup_seconds'] = time.perf_counter()-started
        sys.path.insert(0, str(ROOT/BASE/'eval'))
        from runtime_guards import generate_with_checkpoint
        sampling = SamplingParams(temperature=0., top_p=1., seed=42, max_tokens=128, skip_special_tokens=True)
        started = time.perf_counter()
        outputs = generate_with_checkpoint(llm, prompts, sampling, None, out/'probes.partial.jsonl')
        ledger['generation_seconds'] = time.perf_counter()-started
        records = []
        for mapping, prompt, result in zip(mappings, prompts, outputs, strict=True):
            require(list(result.prompt_token_ids) == prompt['prompt_token_ids'], 'Prompt retokenized or altered')
            output = result.outputs[0]
            ids = list(output.token_ids)
            require(len(ids) <= 128 and output.finish_reason in ['stop', 'length'], 'Incomplete probe')
            records.append(dict(**mapping, text=output.text, token_ids=ids, tokens=len(ids),
                                prompt_tokens=len(result.prompt_token_ids), finish_reason=output.finish_reason,
                                key=boxed_key(output.text)))
        save(out/'probes.json', dict(status='completed_short_probes', records=records))
        ledger.update(status='probes_completed_grading_pending', completed_unix=time.time(),
                      probes_sha256=sha(out/'probes.json'), all_exact_token_prefixes_verified=True,
                      gpu_work_finished=True)
    except BaseException as error:
        ledger.update(status='incomplete', error=repr(error), stopped_unix=time.time())
        raise
    finally:
        watchdog.cancel()
        save(out/'ledger.json', ledger)
    print(json.dumps(ledger))


if __name__ == '__main__':
    main()
