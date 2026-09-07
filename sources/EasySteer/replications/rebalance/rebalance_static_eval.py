"""Paired vLLM evaluation of the static part of ReBalance.

This deliberately excludes ReBalance's confidence/variance controller.  It
transfers only the published direction tensor, the paragraph-boundary trigger,
and a fixed coefficient into EasySteer, so backend/layer/vector behavior can be
validated before dynamic state is added.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from easysteer.vectors import from_pt_direction
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec


DEFAULT_MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATASET = (
    "/root/autodl-tmp/ReBalance/Data/Math_GSM8K_200_seed42/test.jsonl"
)
DEFAULT_VECTOR = (
    "/root/autodl-tmp/ReBalance/vectors/DeepSeek-R1-Distill-Qwen-1.5B/"
    "steer_vector_layer19_conf_mixed.pt"
)
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/results/easysteer/"
    "rebalance_static_vllm_gsm8k20.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--vector", default=DEFAULT_VECTOR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--layer",
        type=int,
        default=18,
        help=(
            "EasySteer decoder-output layer. ReBalance injects before block 19, "
            "which is the output of block 18."
        ),
    )
    parser.add_argument("--scale", type=float, default=-1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Use EasySteer's in-graph direct-steering path instead of eager mode.",
    )
    return parser.parse_args()


def resolve_file(path_text: str, filename: str) -> Path:
    requested = Path(path_text).resolve()
    if requested.is_file():
        return requested
    matches = sorted(Path("/root/autodl-tmp").rglob(filename))
    if len(matches) == 1:
        print(f"Resolved missing default to: {matches[0]}")
        return matches[0]
    rendered = "\n".join(f"  - {path}" for path in matches) or "  (none)"
    raise FileNotFoundError(
        f"Could not resolve {requested}. Candidates named {filename}:\n{rendered}"
    )


def load_examples(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    if offset < 0 or limit <= 0:
        raise ValueError("--offset must be >= 0 and --limit must be > 0")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = rows[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"Requested {limit} examples at offset {offset}, but only "
            f"{len(selected)} are available"
        )
    for index, row in enumerate(selected):
        if "problem" not in row or "answer" not in row:
            raise ValueError(
                f"Dataset row {offset + index} must contain problem and answer"
            )
    return selected


def build_prompt(tokenizer: AutoTokenizer, problem: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Please reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        },
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def grade_answer(gold: str, prediction: str) -> tuple[bool, str | None]:
    try:
        gold_parsed = parse(
            f"${gold}$",
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        prediction_parsed = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
        )
        return bool(verify(gold_parsed, prediction_parsed)), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def generate_records(
    llm: LLM,
    prompts: list[str],
    examples: list[dict[str, Any]],
    params: SamplingParams,
    boundary_ids: set[int],
    steering: SteeringSpec | None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params=params,
        steering=steering,
        use_tqdm=True,
    )
    seconds = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for local_index, (example, request_output) in enumerate(
        zip(examples, outputs, strict=True)
    ):
        output = request_output.outputs[0]
        token_ids = list(output.token_ids)
        correct, grading_error = grade_answer(example["answer"], output.text)
        records.append(
            {
                "dataset_index": local_index,
                "problem": example["problem"],
                "gold": example["answer"],
                "correct": correct,
                "grading_error": grading_error,
                "tokens": len(token_ids),
                "boundary_tokens": sum(token in boundary_ids for token in token_ids),
                "finish_reason": output.finish_reason,
                "text": output.text,
            }
        )
    return records, seconds


def summarize(
    records: list[dict[str, Any]], seconds: float, max_tokens: int
) -> dict[str, Any]:
    counts = [record["tokens"] for record in records]
    total = sum(counts)
    return {
        "examples": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": mean(record["correct"] for record in records),
        "mean_tokens": mean(counts),
        "total_tokens": total,
        "capped": sum(
            record["finish_reason"] == "length" and record["tokens"] >= max_tokens
            for record in records
        ),
        "boundary_tokens": sum(record["boundary_tokens"] for record in records),
        "generation_seconds": seconds,
        "tokens_per_second": total / seconds if seconds else None,
        "grading_errors": sum(record["grading_error"] is not None for record in records),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("--max-model-len must be greater than --max-tokens")

    dataset_path = resolve_file(args.dataset, "test.jsonl")
    vector_path = resolve_file(
        args.vector, "steer_vector_layer19_conf_mixed.pt"
    )
    output_path = Path(args.output).resolve()
    examples = load_examples(dataset_path, args.offset, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    # Match ReBalance's tokenizer scan: any vocabulary item containing the
    # double-newline marker is treated as a reasoning-step boundary.
    newline_ids = sorted(
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if "ĊĊ" in token
    )
    if not newline_ids:
        raise RuntimeError("No ReBalance paragraph-boundary token ids found")

    payload = from_pt_direction(str(vector_path), layers=[args.layer])
    steering = SteeringSpec(
        vectors=[
            VectorSpec(
                name="rebalance_static",
                data=payload,
                algorithm="direct",
                scale=args.scale,
                layers=[args.layer],
                normalize=False,
                apply=ApplySpec(generation_tokens=newline_ids),
            )
        ]
    )
    print(
        f"Preflight: {len(examples)} examples, vector={vector_path}, "
        f"EasySteer layer={args.layer}, boundaries={len(newline_ids)}"
    )

    result: dict[str, Any] = {
        "scope": "ReBalance-static-vllm (no dynamic confidence controller)",
        "limitations": [
            "Fixed coefficient only; confidence and variance state are excluded.",
            "ReBalance gates boundaries to the think phase; current EasySteer "
            "token selector also sees matching boundaries after </think>.",
            "Accuracy uses math-verify and may differ slightly from ReBalance check.py.",
        ],
        "protocol": {
            "model": str(model_path),
            "dataset": str(dataset_path),
            "vector": str(vector_path),
            "offset": args.offset,
            "limit": args.limit,
            "rebalance_source_layer": 19,
            "easysteer_output_layer": args.layer,
            "layer_mapping": "before ReBalance block 19 equals output of block 18",
            "scale": args.scale,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "execution_mode": "in_graph" if args.compiled else "eager",
            "boundary_token_id_count": len(newline_ids),
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
            steer_algorithms=["direct"],
            enforce_eager=not args.compiled,
            steer_graph_mode="in_graph" if args.compiled else "auto",
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            seed=args.seed,
        )
        result["startup_seconds"] = time.perf_counter() - started
        result["environment"]["gpu"] = torch.cuda.get_device_name(0)

        prompts = [build_prompt(tokenizer, example["problem"]) for example in examples]
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            skip_special_tokens=True,
        )
        boundary_set = set(newline_ids)

        print("Running vLLM baseline...")
        baseline, baseline_seconds = generate_records(
            llm, prompts, examples, params, boundary_set, steering=None
        )
        result["baseline"] = {
            "summary": summarize(baseline, baseline_seconds, args.max_tokens),
            "records": baseline,
        }
        write_result(output_path, result)
        print(f"Baseline checkpoint: {output_path}")

        print("Running ReBalance static steering...")
        static, static_seconds = generate_records(
            llm, prompts, examples, params, boundary_set, steering=steering
        )
        result["rebalance_static"] = {
            "summary": summarize(static, static_seconds, args.max_tokens),
            "records": static,
        }

        improved = [
            args.offset + index
            for index, (base, steered) in enumerate(zip(baseline, static, strict=True))
            if not base["correct"] and steered["correct"]
        ]
        degraded = [
            args.offset + index
            for index, (base, steered) in enumerate(zip(baseline, static, strict=True))
            if base["correct"] and not steered["correct"]
        ]
        base_summary = result["baseline"]["summary"]
        static_summary = result["rebalance_static"]["summary"]
        result["comparison"] = {
            "mean_token_change": static_summary["mean_tokens"]
            - base_summary["mean_tokens"],
            "mean_token_change_percent": (
                static_summary["mean_tokens"] / base_summary["mean_tokens"] - 1
            )
            * 100,
            "accuracy_change_percentage_points": (
                static_summary["accuracy"] - base_summary["accuracy"]
            )
            * 100,
            "improved_indices": improved,
            "degraded_indices": degraded,
        }
        write_result(output_path, result)

        print("\n===== REBALANCE STATIC VLLM SUMMARY =====")
        print(f"Examples: {args.limit} (offset {args.offset})")
        print(
            f"Baseline: acc={base_summary['accuracy']:.3f}, "
            f"mean_tokens={base_summary['mean_tokens']:.1f}, "
            f"capped={base_summary['capped']}, "
            f"time={base_summary['generation_seconds']:.1f}s"
        )
        print(
            f"Static:   acc={static_summary['accuracy']:.3f}, "
            f"mean_tokens={static_summary['mean_tokens']:.1f}, "
            f"capped={static_summary['capped']}, "
            f"time={static_summary['generation_seconds']:.1f}s"
        )
        print(f"Token change: {result['comparison']['mean_token_change_percent']:+.1f}%")
        print(
            "Accuracy change: "
            f"{result['comparison']['accuracy_change_percentage_points']:+.1f} pp"
        )
        print(f"Improved/degraded: {len(improved)}/{len(degraded)}")
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
