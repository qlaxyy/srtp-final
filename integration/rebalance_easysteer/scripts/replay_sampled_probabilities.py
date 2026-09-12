"""Recover raw modal and selected probabilities on saved unsteered prefixes."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np

from mechanism_candidates import ROOT, read, save, sha, require
from prepare_sampled_probability_replay import prompt_body


def validate(bundle, raw_path, model_check=False):
    plan = read(bundle/'plan.json'); rows = read(raw_path)['records']; questions = read(bundle/'questions.json')
    require(plan['status'] == 'prepared_saved100_probability_replay_no_generation', 'Wrong plan')
    require(sha(raw_path) == plan['source_answer_sha256'], 'Saved answer hash changed')
    require(sha(bundle/'questions.json') == plan['question_mapping_sha256'], 'Mapping changed')
    require(len(rows) == len(questions) == plan['questions'] == 100, 'Wrong support')
    for row, q in zip(rows, questions, strict=True):
        require(hashlib.sha256(json.dumps(row['token_ids'], separators=(',', ':')).encode()).hexdigest() == q['token_ids_sha256'], 'Token IDs changed')
        require(0 < q['thinking_tokens'] <= len(row['token_ids']) <= 16000, 'Invalid length')
    for p, digest in plan['source_sha256'].items(): require(sha(ROOT/p, source=True) == digest, 'Source changed: '+p)
    if model_check:
        for p, digest in plan['model_files_sha256'].items(): require(sha(Path(plan['model'])/p) == digest, 'Model changed: '+p)
    return plan, rows, questions


def prompt_function(plan):
    text = (ROOT/plan['prompt_builder_source']).read_text(encoding='utf-8')
    require(hashlib.sha256(prompt_body(text).encode()).hexdigest() == plan['prompt_builder_AST_sha256'], 'Prompt builder changed')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt')
    namespace = dict(AutoTokenizer=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<verified_prompt_builder>', 'exec'), namespace)
    return namespace['build_prompt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-inputs', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); bundle = a.bundle.resolve(); initial = read(bundle/'plan.json')
    raw_path = Path(initial['remote_source_answer']) if os.name != 'nt' else ROOT/initial['local_source_answer']
    plan, rows, questions = validate(bundle, raw_path, model_check=bool(a.prepare_inputs or a.execute))
    if a.prepare_inputs:
        require(not a.execute and not a.prepare_inputs.exists(), 'Fresh CPU input preparation required')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        build_prompt = prompt_function(plan); prompts = []
        for row, q in zip(rows, questions, strict=True):
            text = build_prompt(tokenizer, row['problem']); ids = tokenizer.encode(text)
            require(tokenizer.decode(ids, skip_special_tokens=False) == text, 'Prompt roundtrip changed')
            require(ids and len(ids)+q['thinking_tokens'] <= 32768, 'Prompt length overflow')
            prompts.append(dict(index=q['index'], prompt_token_ids=ids, prompt_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        save(a.prepare_inputs, dict(plan_sha256=sha(bundle/'plan.json'), tokenizer_sha256=plan['model_files_sha256']['tokenizer.json'], prompts=prompts, model_loads=0, CUDA_calls=0))
        print(dict(status='CPU_tokenizer_inputs_prepared', input_sha256=sha(a.prepare_inputs), max_prompt_tokens=max(len(x['prompt_token_ids']) for x in prompts)))
        return
    if not a.execute:
        print(dict(status='local_CPU_preflight_no_model_import', questions=100, plan_sha256=sha(bundle/'plan.json'), new_answers=0)); return
    require(os.name != 'nt' and a.output and not a.output.exists() and a.inputs, 'Fresh Linux replay required')
    inputs = read(a.inputs); require(inputs['plan_sha256'] == sha(bundle/'plan.json'), 'Prepared input plan changed')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Dirty source')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Other GPU process')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    out = a.output.resolve(); out.mkdir(parents=True); save(out/'plan.json', plan); save(out/'inputs.json', inputs)
    ledger = dict(status='loading', started_unix=time.time(), plan_sha256=sha(bundle/'plan.json'), input_sha256=sha(a.inputs),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, transformers=transformers.__version__, completed_questions=0, new_answers=0, files_sha256={})
    save(out/'ledger.json', ledger)
    try:
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(plan['model'], torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).cuda().eval()
        torch.cuda.synchronize(); ledger['startup_seconds'] = time.perf_counter()-start
        start = time.perf_counter(); input_tokens = 0
        for row, q, prompt in zip(rows, questions, inputs['prompts'], strict=True):
            require(time.time()-ledger['started_unix'] < plan['runtime']['maximum_seconds_including_loading'], 'Replay deadline')
            require(prompt['index'] == q['index'], 'Prepared prompt order changed')
            token_ids = row['token_ids'][:q['thinking_tokens']]; prompt_ids = prompt['prompt_token_ids']; count = len(token_ids)
            ids = prompt_ids+token_ids; input_tokens += len(ids)
            maximum = np.empty(count, dtype=np.float32); selected = np.empty_like(maximum); argmax = np.empty(count, dtype=np.int32)
            with torch.inference_mode():
                states = model.model(torch.tensor([ids], device='cuda'), use_cache=False, return_dict=True).last_hidden_state
                # State P-1 predicts output token0; the last input state is unused.
                prediction_states = states[0, len(prompt_ids)-1:len(prompt_ids)+count-1]
                require(prediction_states.shape[0] == count, 'Prediction shift mismatch')
                for begin in range(0, count, plan['runtime']['LM_head_chunk_tokens']):
                    end = min(count, begin+plan['runtime']['LM_head_chunk_tokens'])
                    logits = model.lm_head(prediction_states[begin:end]).float()
                    normalizer = torch.logsumexp(logits, dim=-1)
                    top, top_id = logits.max(dim=-1)
                    target = torch.tensor(token_ids[begin:end], device='cuda')
                    chosen = logits.gather(1, target[:, None])[:, 0]
                    maximum[begin:end] = (top-normalizer).exp().cpu().numpy()
                    selected[begin:end] = (chosen-normalizer).exp().cpu().numpy()
                    argmax[begin:end] = top_id.cpu().numpy()
                del states, prediction_states, logits
            require(np.isfinite(maximum).all() and np.isfinite(selected).all() and np.all((selected >= 0)&(maximum <= 1)&(selected <= maximum)), 'Invalid probabilities')
            same = argmax == np.array(token_ids); require(np.array_equal(maximum[same], selected[same]), 'Greedy identity failed')
            file = f"q{q['index']:03d}.npz"
            np.savez_compressed(out/file, maximum=maximum, selected=selected, argmax=argmax, token_ids=np.array(token_ids, dtype=np.int32))
            ledger['files_sha256'][file] = sha(out/file)
            ledger.update(status='replaying', completed_questions=q['index']+1, replay_seconds=time.perf_counter()-start, input_tokens=input_tokens)
            save(out/'ledger.json', ledger)
            if (q['index']+1)%10 == 0: print(f"Saved probability replay {q['index']+1}/100; {ledger['replay_seconds']:.1f}s", flush=True)
        del model; torch.cuda.empty_cache()
        ledger.update(status='completed', completed_unix=time.time(), gpu_work_finished=True)
        save(out/'ledger.json', ledger); print(json.dumps(ledger), flush=True)
    except BaseException as error:
        ledger.update(status='incomplete', error=repr(error), stopped_unix=time.time()); save(out/'ledger.json', ledger); raise


if __name__ == '__main__': main()
