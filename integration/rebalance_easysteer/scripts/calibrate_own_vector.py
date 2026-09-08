"""Independent MATH-train calibration; no benchmark labels enter fitting.

The released extraction rules are imported unchanged. Generation is accelerated
with vLLM greedy decoding. Hidden states are replayed with Transformers SDPA,
retaining only author hidden-state index 19 at the author's step positions.
Offline confidence intentionally follows the released geometric/re-tokenized
estimator; online control remains the released arithmetic-mean adaptation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
ROOT = Path(__file__).resolve().parents[3]
MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def save(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prompt(tokenizer, problem):
    return tokenizer.apply_chat_template([
        {"role": "system", "content":
         "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ], tokenize=False, add_generation_prompt=True)


def generate(out, model_path=MODEL):
    if (out / "generations.jsonl").exists():
        raise FileExistsError("Generation already exists; do not repeat it")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    source = ROOT / "sources/ReBalance/Data/Math_Train/test.jsonl"
    train = read(source)
    test = read(ROOT / "sources/ReBalance/Data/Math_Math500/test.jsonl")
    normalize = lambda x: re.sub(r"\s+", "", x)
    forbidden = {normalize(x["problem"]) for x in test}
    eligible = [i for i, x in enumerate(train)
                if normalize(x["problem"]) not in forbidden]
    selected = random.Random(42).sample(eligible, 500)
    selected_problems = {normalize(train[i]["problem"]) for i in selected}
    assert len(selected_problems) == 500
    split_check = dict(train_total=len(train), selected_indices=len(selected),
                       unique_selected_problems=len(selected_problems))
    for name in ("Math_Math500", "Math_GSM8K"):
        held_out = read(ROOT / f"sources/ReBalance/Data/{name}/test.jsonl")
        overlap = selected_problems & {normalize(x["problem"]) for x in held_out}
        assert not overlap, f"Calibration overlaps {name}"
        split_check[name] = dict(test_count=len(held_out),
                                exact_normalized_problem_overlap=len(overlap))
    save(out / "split_check.json", split_check)
    manifest = dict(seed=42, count=500, train_indices=selected,
        train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        excluded_test_overlap=len(train)-len(eligible),
        model=str(model_path), max_tokens=16000, temperature=0,
        generation="vLLM greedy, raw selected-token log probabilities",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip())
    save(out / "manifest.json", manifest)
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=str(model_path), dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=.9, seed=42,
              enable_prefix_caching=False, enable_chunked_prefill=False)
    started = time.time()
    results = llm.generate([prompt(tok, train[i]["problem"]) for i in selected],
        SamplingParams(temperature=0, max_tokens=16000, logprobs=1,
                       seed=42, skip_special_tokens=True))
    with (out / "generations.jsonl").open("x") as f:
        for index, result in zip(selected, results, strict=True):
            seq = result.outputs[0]
            row = dict(train_index=index, problem=train[index]["problem"],
                prompt_token_ids=list(result.prompt_token_ids),
                token_ids=list(seq.token_ids), text=seq.text,
                logprobs=[p[t].logprob for t, p in
                          zip(seq.token_ids, seq.logprobs, strict=True)],
                finish_reason=seq.finish_reason)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(out / "generation_summary.json", dict(count=len(results),
        seconds=time.time()-started,
        mean_tokens=sum(len(x.outputs[0].token_ids) for x in results)/len(results),
        capped=sum(x.outputs[0].finish_reason=="length" for x in results)))


def extract(out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    hidden_dir = out / "hidden"
    hidden_dir.mkdir(exist_ok=True)
    rows = read(out / "generations.jsonl")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == manifest["count"] == 500
    assert [r["train_index"] for r in rows] == manifest["train_indices"]
    for row in rows:
        assert len(row["token_ids"]) == len(row["logprobs"]) > 0
        assert all(math.isfinite(p) and p <= 1e-5 for p in row["logprobs"])
    records = []
    capture = {}
    positions = []
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        capture["h"] = h[0, positions].detach().cpu()
    handle = model.model.layers[18].register_forward_hook(hook)
    started = time.time()
    # One forward-only check, with saved calibration input, verifies that the
    # memory-saving hook is exactly the author hidden_states[19] representation.
    check_ids = tok(prompt(tok, rows[0]["problem"]), return_tensors="pt")["input_ids"][:, :128]
    positions = [0, check_ids.shape[1]-1]
    with torch.inference_mode():
        check = model.model(check_ids.cuda(), use_cache=False,
                            output_hidden_states=True, return_dict=True)
    reference = check.hidden_states[19][0, positions].cpu()
    assert torch.equal(reference, capture.pop("h"))
    save(out / "hidden_hook_check.json", dict(equal=True, max_abs_error=0,
        author_hidden_index=19, decoder_output_layer=18,
        tokens=check_ids.shape[1], positions=positions))
    del check, reference
    for i, row in enumerate(rows):
        # Exact released text-space re-tokenization, including its limitations.
        ids = tok(prompt(tok, row["problem"]) + row["text"],
                  return_tensors="pt")["input_ids"]
        flat = ids[0].tolist()
        close = tok.encode("</think>", add_special_tokens=False)
        end = next((j for j in range(len(flat)-len(close)+1)
                    if flat[j:j+len(close)] == close), len(flat))
        tokens = tok.convert_ids_to_tokens(flat)
        positions = [j+1 for j in range(end-1) if "ĊĊ" in tokens[j]]
        path = hidden_dir / f"hidden_{i}.pt"
        if not path.exists():
            with torch.inference_mode():
                model.model(ids.cuda(), use_cache=False, return_dict=True)
            torch.save({19: {0: {"step": capture.pop("h")}}}, path)
        confidences = []
        start = 0
        for seg in re.split(r"\n\n+", row["text"].split("</think>")[0]):
            stop = start + len(tok(seg, add_special_tokens=False)["input_ids"])
            if stop > start:
                assert stop <= len(row["logprobs"]), (i, start, stop)
                confidences.append(math.exp(sum(row["logprobs"][start:stop]) /
                                            (stop-start)))
            start = stop
        records.append(dict(idx=i, train_index=row["train_index"],
            generated_responses=[row["text"]], sentence_confidences=confidences,
            step_count=len(positions)))
        if (i+1) % 25 == 0:
            print(f"Hidden extraction {i+1}/500, {time.time()-started:.1f}s", flush=True)
    handle.remove()
    with (out / "calibration.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(out / "extraction_summary.json", dict(seconds=time.time()-started,
        count=len(records), matching_author_offset=sum(
            r["step_count"] == len(r["sentence_confidences"])-1 for r in records)))


def fit(out):
    import torch
    torch.set_num_threads(8)
    source = ROOT / "sources/ReBalance/hidden_analysis_auto.py"
    spec = importlib.util.spec_from_file_location("author_extract", source)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    data = str(out / "calibration.jsonl")
    q25, q75, n = author.compute_confidence_quartiles(data)
    v25, v75, nv = author.compute_global_diff_quartiles(data, expected_offset=1)
    assert n > 0 and nv > 0 and q25 < q75 and v25 < v75
    ds = author.batch_build_all_mixed(19, data, str(out / "hidden"),
        threshold=q25, high_threshold=q75, max_files=500, expected_offset=1)
    vector, positive, negative = author.build_steer_vector_mean_only(ds)
    lda = author.compute_lda_separator_and_steer(ds)
    assert lda["ok"] and 0 < lda["alpha_mean_S"] <= lda["alpha_all_S"]
    assert torch.isfinite(vector).all()
    torch.save(vector, out / "own_vector_layer19.pt")
    params = dict(q25c=q25, q75c=q75, q25v=v25, q75v=v75,
        low_val_1=-lda["alpha_mean_S"], low_val_2=-lda["alpha_all_S"],
        high_val_2=.1, initial_coef=-1.)
    runtime_path = ROOT / "sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py"
    runtime_spec = importlib.util.spec_from_file_location("calibration_runtime", runtime_path)
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)
    control = runtime.ReBalanceParams(boundary_token_ids=(1,),
        think_start_token_id=2, think_end_token_id=3, **params)
    c, v = torch.meshgrid(torch.linspace(0, 1, 501),
                         torch.linspace(0, .25, 251), indexing="ij")
    values = runtime.compute_rebalance_coefficient(c, v, control)
    assert torch.isfinite(values).all()
    assert values.min() >= params["low_val_2"] - 1e-5
    assert values.max() <= .1 + 1e-5
    anchors = runtime.compute_rebalance_coefficient(
        torch.tensor([q25, 1.]), torch.tensor([v75, v25]), control)
    assert torch.allclose(anchors, torch.tensor([params["low_val_2"], .1]), atol=2e-5)
    save(out / "curve_check.json", dict(grid_points=values.numel(), finite=True,
        min=values.min().item(), max=values.max().item(),
        low_high_anchors=anchors.tolist()))
    save(out / "fit.json", dict(parameters=params, lda=lda,
        vector_norm=vector.norm().item(), positives=positive, negatives=negative,
        usable_questions=len(ds.datasets), confidence_count=n, variance_count=nv,
        vector_sha256=hashlib.sha256((out / "own_vector_layer19.pt").read_bytes()).hexdigest(),
        author_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method="Released mixed-label/raw mean-difference vector and diagonal LDA. "
        "Our explicit endpoint mapping: low1=-alpha_mean_S; low2=-alpha_all_S. "
        "high2=0.1 and initial=-1 are fixed design constants, not fitted. "
        "Offline geometric confidence; online released arithmetic controller. "
        "Fixed author layer19; no layer search or benchmark tuning.",
        prompt_boundary_correction=(out / "prompt_boundary_correction.json").exists()))
    print((out / "fit.json").read_text(), flush=True)


def align(out):
    """Remove prompt boundaries from saved author-format states; no forward pass.

    The released extractor scans prompt + response for blank lines. Its label
    builder only counts response steps, silently dropping multi-paragraph
    questions. Correct the scope, retaining the released response-step offset.
    """
    import torch
    from transformers import AutoTokenizer
    source = out.parent
    assert (source / "calibration.jsonl").exists()
    tok = AutoTokenizer.from_pretrained(MODEL)
    generations = read(source / "generations.jsonl")
    records = read(source / "calibration.jsonl")
    (out / "hidden").mkdir(exist_ok=True)
    corrections = []
    for i, (row, gen) in enumerate(zip(records, generations, strict=True)):
        # Re-tokenize exactly as the author extractor, so prompt token indices
        # match the saved full-sequence hidden states.
        ids = tok(prompt(tok, gen["problem"]))["input_ids"]
        extra = sum("ĊĊ" in t for t in tok.convert_ids_to_tokens(ids))
        raw = torch.load(source / "hidden" / f"hidden_{i}.pt", weights_only=False)
        values = raw[19][0]["step"]
        corrected = values[extra:].clone()
        assert corrected.shape[0] == len(row["sentence_confidences"])-1, i
        assert torch.equal(corrected, values[extra:])
        torch.save({19: {0: {"step": corrected}}}, out / "hidden" / f"hidden_{i}.pt")
        row["step_count"] = corrected.shape[0]
        if extra:
            corrections.append(dict(index=i, train_index=row["train_index"],
                                    prompt_boundaries_removed=extra))
    for name in ("manifest.json", "generations.jsonl", "generation_summary.json",
                 "split_check.json", "hidden_hook_check.json", "preflight_checks.json"):
        if not (out / name).exists():
            os.link(source / name, out / name)
    with (out / "calibration.jsonl").open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(out / "prompt_boundary_correction.json", dict(source=str(source),
        corrected_questions=len(corrections), matching_questions=len(records),
        regenerated_answers=0, extra_model_forwards=0, corrections=corrections))
    save(out / "extraction_summary.json", dict(count=len(records),
        matching_author_offset=len(records), corrected_prompt_boundaries=True))
    print(f"Aligned {len(records)} questions; corrected {len(corrections)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "extract", "align", "fit"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args.output)
