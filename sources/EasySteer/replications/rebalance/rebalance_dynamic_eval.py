"""Paired GSM8K evaluation of author-code ReBalance on EasySteer vLLM."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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
    parser.add_argument("--initial-coef", type=float, default=-1.0)
    parser.add_argument("--q25c", type=float, default=0.662293)
    parser.add_argument("--q75c", type=float, default=0.94805)
    parser.add_argument("--low-val-1", type=float, default=-1.02)
    parser.add_argument("--q25v", type=float, default=0.000560)
    parser.add_argument("--q75v", type=float, default=0.011597)
    parser.add_argument("--low-val-2", type=float, default=-1.91)
    parser.add_argument("--high-val-2", type=float, default=0.1)
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
    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    dataset_path = resolve_file(args.dataset, "test.jsonl")
    vector_path = resolve_file(
        args.vector, "steer_vector_layer19_conf_mixed.pt"
    )
    output_path = Path(args.output).resolve()
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
    }
    payload = from_pt_direction(str(vector_path), layers=[args.layer])
    steering = SteeringSpec(
        vectors=[
            VectorSpec(
                name="rebalance_dynamic",
                data=payload,
                algorithm="rebalance",
                scale=1.0,
                layers=[args.layer],
                normalize=False,
                apply=ApplySpec(generation_tokens=boundary_ids),
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

        print("Running paired vLLM baseline...")
        baseline, baseline_seconds = generate_records(
            llm, prompts, examples, sampling, boundary_set, steering=None
        )
        baseline_summary = summarize(
            baseline, baseline_seconds, args.max_tokens
        )
        result["baseline"] = {
            "summary": baseline_summary,
            "records": baseline,
        }
        write_result(output_path, result)

        print("Running dynamic ReBalance steering...")
        dynamic, dynamic_seconds = generate_records(
            llm, prompts, examples, sampling, boundary_set, steering=steering
        )
        dynamic_summary = summarize(dynamic, dynamic_seconds, args.max_tokens)
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
