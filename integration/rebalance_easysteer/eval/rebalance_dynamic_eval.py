"""Paired GSM8K evaluation of author-code ReBalance on EasySteer vLLM."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import hashlib
import signal
import subprocess
import threading
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from easysteer.vectors import from_pt_direction
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

from rebalance_static_eval import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    DEFAULT_VECTOR,
    build_prompt,
    generate_records,
    load_examples,
    resolve_file,
    summarize,
    write_result,
)


DEFAULT_OUTPUT = (
    "/root/autodl-tmp/results/easysteer/"
    "rebalance_dynamic_vllm_gsm8k20.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--vector", default=DEFAULT_VECTOR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-timeout-seconds", type=int, default=1500)
    parser.add_argument("--initial-coef", type=float, default=-1.0)
    parser.add_argument("--q25c", type=float, default=0.662293)
    parser.add_argument("--q75c", type=float, default=0.94805)
    parser.add_argument("--low-val-1", type=float, default=-1.02)
    parser.add_argument("--q25v", type=float, default=0.000560)
    parser.add_argument("--q75v", type=float, default=0.011597)
    parser.add_argument("--low-val-2", type=float, default=-1.91)
    parser.add_argument("--high-val-2", type=float, default=0.1)
    parser.add_argument("--curve-tau", type=float, default=0.01)
    parser.add_argument("--calibration-fit", type=Path)
    parser.add_argument("--paper-fit", type=Path)
    parser.add_argument("--baseline-result", type=Path,
                        help="Reuse a compatible saved baseline; no generation repeat")
    return parser.parse_args()


def single_token_id(tokenizer: AutoTokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"Tokenizer produced no ids for {text!r}")
    if len(token_ids) != 1:
        print(
            f"Warning: {text!r} maps to {token_ids}; matching author code by "
            "using the first id"
        )
    return token_ids[0]


def comparison(
    baseline: list[dict[str, Any]],
    dynamic: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    dynamic_summary: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    improved = [
        offset + index
        for index, (base, steered) in enumerate(
            zip(baseline, dynamic, strict=True)
        )
        if not base["correct"] and steered["correct"]
    ]
    degraded = [
        offset + index
        for index, (base, steered) in enumerate(
            zip(baseline, dynamic, strict=True)
        )
        if base["correct"] and not steered["correct"]
    ]
    return {
        "mean_token_change": (
            dynamic_summary["mean_tokens"] - baseline_summary["mean_tokens"]
        ),
        "mean_token_change_percent": (
            dynamic_summary["mean_tokens"] / baseline_summary["mean_tokens"] - 1
        )
        * 100,
        "accuracy_change_percentage_points": (
            dynamic_summary["accuracy"] - baseline_summary["accuracy"]
        )
        * 100,
        "improved_indices": improved,
        "degraded_indices": degraded,
    }


def main() -> None:
    args = parse_args()
    if args.paper_fit and args.calibration_fit:
        raise ValueError("Select one baseline definition")
    paper = None
    if args.paper_fit:
        paper = json.loads(args.paper_fit.read_text(encoding="utf-8"))
        args.layer = paper["decoder_output_layer"]
        for key in ("q25c", "q75c", "q25v", "q75v", "initial_coef"):
            setattr(args, key, paper["parameters"][key])
    fitted = None
    if args.calibration_fit:
        fitted = json.loads(args.calibration_fit.read_text(encoding="utf-8"))
        expected = {"q25c", "q75c", "q25v", "q75v", "low_val_1",
                    "low_val_2", "high_val_2", "initial_coef"}
        if set(fitted["parameters"]) not in (expected, expected | {"curve_tau"}):
            raise ValueError("Calibration must provide every controller parameter")
        for key, value in fitted["parameters"].items():
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid fitted parameter: {key}")
            setattr(args, key, value)
        if fitted.get("version") == "auto-code-v2":
            if Path(fitted["model"]).resolve() != Path(args.model).resolve():
                raise ValueError("Automatic calibration model mismatch")
            args.layer = fitted["decoder_output_layer"]
    if paper is None:
        from vllm.steer_vectors.rebalance import validate_curve_targets
        validate_curve_targets(args.q25c, args.q75c, args.low_val_1, args.curve_tau)
    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    dataset_path = resolve_file(args.dataset, "test.jsonl")
    vector_path = resolve_file(
        args.vector, "steer_vector_layer19_conf_mixed.pt"
    )
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    examples = load_examples(dataset_path, args.offset, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    boundary_ids = sorted(
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if "ĊĊ" in token
    )
    if not boundary_ids:
        raise RuntimeError("No ReBalance paragraph-boundary token ids found")
    think_start_id = single_token_id(tokenizer, "<think>")
    think_end_id = single_token_id(tokenizer, "</think>")

    dynamic_params = {
        "boundary_token_ids": boundary_ids,
        "think_start_token_id": think_start_id,
        "think_end_token_id": think_end_id,
        "initial_coef": args.initial_coef,
        "q25c": args.q25c,
        "q75c": args.q75c,
        "low_val_1": args.low_val_1,
        "q25v": args.q25v,
        "q75v": args.q75v,
        "low_val_2": args.low_val_2,
        "high_val_2": args.high_val_2,
        "curve_tau": args.curve_tau,
    }
    payload = from_pt_direction(str(vector_path), layers=[args.layer])
    if paper is not None:
        dynamic_params["paper_parameters"] = paper["parameters"]["paper_parameters"]
    steering = SteeringSpec(
        vectors=[
            VectorSpec(
                name="rebalance_dynamic",
                data=payload,
                algorithm="rebalance",
                scale=1.0,
                layers=[args.layer],
                normalize=False,
                apply=(ApplySpec(generation="all") if paper is not None else
                       ApplySpec(generation_tokens=boundary_ids)),
                params=dynamic_params,
            )
        ]
    )
    result: dict[str, Any] = {
        "scope": "ReBalance-official-code-vllm",
        "confidence_definition": (
            "Arithmetic mean of raw tokenwise maximum probabilities, matching "
            "the released Qwen2 dynamic implementation. The paper's geometric "
            "mean is reserved for ReBalance-paper-faithful-vllm."
        ),
        "protocol": {
            "model": str(model_path),
            "dataset": str(dataset_path),
            "vector": str(vector_path),
            "offset": args.offset,
            "limit": args.limit,
            "rebalance_source_layer": 19,
            "easysteer_output_layer": args.layer,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "execution_mode": "in_graph",
            "dynamic_params": dynamic_params,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
    }
    root = Path(__file__).resolve().parents[3]
    result["provenance"] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
    }
    result["protocol"]["group_timeout_seconds"] = args.group_timeout_seconds
    if paper is not None:
        if result["provenance"]["vector_sha256"] != paper["vector_sha256"]:
            raise ValueError("Paper vector hash mismatch")
        if str(model_path) != paper["model"]:
            raise ValueError("Paper calibration model mismatch")
        result["scope"] = "ReBalance-paper-reconstruction-vllm-v1"
        result["confidence_definition"] = "Geometric mean of raw max probabilities"
        result["protocol"]["rebalance_source_layer"] = paper["hidden_state_index"]
        result["paper_reconstruction"] = paper
        result["provenance"]["paper_fit_sha256"] = hashlib.sha256(
            args.paper_fit.read_bytes()).hexdigest()
    if fitted is not None:
        if result["provenance"]["vector_sha256"] != fitted["vector_sha256"]:
            raise ValueError("Vector does not match calibration fit")
        result["scope"] = "ReBalance-self-calibrated-author-code-vllm"
        if fitted.get("version") == "auto-code-v2":
            result["scope"] = "ReBalance-auto-calibrated-code-v2-vllm"
            result["confidence_definition"] = "Arithmetic mean of raw max probabilities, offline and online"
            result["protocol"]["rebalance_source_layer"] = fitted["hidden_state_index"]
        result["calibration"] = fitted
        result["provenance"]["calibration_fit_sha256"] = hashlib.sha256(
            args.calibration_fit.read_bytes()).hexdigest()

    reused_baseline = None
    if args.baseline_result:
        saved = json.loads(args.baseline_result.read_text(encoding="utf-8"))
        for key in ("model", "dataset", "offset", "limit", "max_tokens",
                    "max_model_len", "temperature", "top_p", "seed", "execution_mode"):
            if saved["protocol"][key] != result["protocol"][key]:
                raise ValueError(f"Incompatible saved baseline protocol: {key}")
        if saved["provenance"]["dataset_sha256"] != result["provenance"]["dataset_sha256"]:
            raise ValueError("Incompatible saved baseline dataset contents")
        for key in ("torch", "vllm"):
            if saved["environment"][key] != result["environment"][key]:
                raise ValueError(f"Incompatible saved baseline environment: {key}")
        reused_baseline = saved["baseline"]
        if len(reused_baseline["records"]) != len(examples):
            raise ValueError("Saved baseline record count mismatch")
        if any(a["problem"] != b["problem"] for a, b in
               zip(reused_baseline["records"], examples, strict=True)):
            raise ValueError("Saved baseline problem order mismatch")
        result["provenance"]["reused_baseline"] = {
            "path": str(args.baseline_result),
            "sha256": hashlib.sha256(args.baseline_result.read_bytes()).hexdigest(),
            "provenance": saved["provenance"],
            "timing_is_historical": True,
        }

    def run_group(prompts, sampling, boundary_set, steering):
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation group exceeded its time budget")
        previous = signal.signal(signal.SIGALRM, timeout_handler)
        started = time.perf_counter()
        # Graders may use SIGALRM internally; keep an independent wall-time cap.
        def enforce_deadline():
            print("Evaluation group exceeded its wall-time budget", file=sys.stderr, flush=True)
            os._exit(124)
        watchdog = threading.Timer(args.group_timeout_seconds, enforce_deadline)
        watchdog.daemon = True
        watchdog.start()
        signal.alarm(args.group_timeout_seconds)
        try:
            records, seconds = generate_records(
                llm, prompts, examples, sampling, boundary_set, steering=steering
            )
            for record in records:
                ids = record["token_ids"]
                ended = think_end_id in ids
                # The chat prompt already opens <think>; delimiter is excluded.
                stop = ids.index(think_end_id) if ended else len(ids)
                record["thinking_tokens"] = stop
                record["thinking_ended"] = ended
                record["answer_tokens"] = len(ids) - stop - int(ended)
            summary = summarize(records, seconds, args.max_tokens)
            summary["mean_thinking_tokens"] = sum(r["thinking_tokens"] for r in records) / len(records)
            summary["mean_answer_tokens"] = sum(r["answer_tokens"] for r in records) / len(records)
            summary["thinking_not_ended"] = sum(not r["thinking_ended"] for r in records)
            summary["group_seconds_including_grading"] = time.perf_counter() - started
            return records, summary
        finally:
            watchdog.cancel()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    llm = None
    try:
        started = time.perf_counter()
        llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_steer_vector=True,
            steer_algorithms=["rebalance"],
            enforce_eager=False,
            steer_graph_mode="in_graph",
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            seed=args.seed,
        )
        result["startup_seconds"] = time.perf_counter() - started
        result["environment"]["gpu"] = torch.cuda.get_device_name(0)
        prompts = [build_prompt(tokenizer, row["problem"]) for row in examples]
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            skip_special_tokens=True,
        )
        boundary_set = set(boundary_ids)

        if reused_baseline is None:
            print("Running paired vLLM baseline...")
            baseline, baseline_summary = run_group(
                prompts, sampling, boundary_set, steering=None
            )
        else:
            print("Reusing verified saved baseline; no baseline generation")
            baseline = reused_baseline["records"]
            baseline_summary = reused_baseline["summary"]
        result["baseline"] = {
            "summary": baseline_summary,
            "records": baseline,
        }
        write_result(output_path, result)

        print("Running dynamic ReBalance steering...")
        dynamic, dynamic_summary = run_group(
            prompts, sampling, boundary_set, steering=steering
        )
        result["rebalance_dynamic"] = {
            "summary": dynamic_summary,
            "records": dynamic,
        }
        result["comparison"] = comparison(
            baseline,
            dynamic,
            baseline_summary,
            dynamic_summary,
            args.offset,
        )
        write_result(output_path, result)

        change = result["comparison"]
        print("\n===== REBALANCE DYNAMIC VLLM SUMMARY =====")
        print(f"Examples: {args.limit} (offset {args.offset})")
        for label, summary in (
            ("Baseline", baseline_summary),
            ("Dynamic", dynamic_summary),
        ):
            print(
                f"{label}: acc={summary['accuracy']:.3f}, "
                f"mean_tokens={summary['mean_tokens']:.1f}, "
                f"capped={summary['capped']}, "
                f"time={summary['generation_seconds']:.1f}s"
            )
        print(f"Token change: {change['mean_token_change_percent']:+.1f}%")
        print(
            "Accuracy change: "
            f"{change['accuracy_change_percentage_points']:+.1f} pp"
        )
        print(
            f"Improved/degraded: {len(change['improved_indices'])}/"
            f"{len(change['degraded_indices'])}"
        )
        print(f"Result file: {output_path}")
    finally:
        del llm
        gc.collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
