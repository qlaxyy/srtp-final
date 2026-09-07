"""Paired MATH-500 evaluation for the EasySteer SEAL replication.

The official EasySteer notebook reports mean generated length on the first
100 MATH-500 examples.  This script keeps that protocol and additionally
records answer accuracy, paired per-example outputs, finish reasons, boundary
hits, and end-to-end generation throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec


DEFAULT_MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATASET = "/root/autodl-tmp/datasets/math500/test.jsonl"
DEFAULT_OUTPUT = "/root/autodl-tmp/results/easysteer/seal_math500_eval.json"
DATASET_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
    "resolve/main/test.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--vector-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def ensure_dataset(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading MATH-500 to {path}")
    try:
        urllib.request.urlretrieve(DATASET_URL, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_examples(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    if offset < 0 or limit <= 0:
        raise ValueError("--offset must be >= 0 and --limit must be > 0")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = rows[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"Requested {limit} examples at offset {offset}, but only "
            f"{len(selected)} are available"
        )
    required = {"problem", "answer", "unique_id"}
    for index, row in enumerate(selected):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Dataset row {offset + index} lacks fields: {sorted(missing)}")
    return selected


def build_prompt(problem: str) -> str:
    return (
        "Please reason step by step, and put your final answer within "
        f"\\boxed{{}}.\nUser: {problem}\nAssistant: <think>"
    )


def grade_answer(gold: str, prediction: str) -> tuple[bool, str | None]:
    try:
        gold_parsed = parse(
            f"${gold}$",
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        prediction_parsed = parse(
            prediction,
            extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        )
        return bool(verify(gold_parsed, prediction_parsed)), None
    except Exception as exc:  # Preserve the run even if one expression is pathological.
        return False, f"{type(exc).__name__}: {exc}"


def summarize(records: list[dict[str, Any]], seconds: float, max_tokens: int) -> dict[str, Any]:
    token_counts = [record["tokens"] for record in records]
    total_tokens = sum(token_counts)
    return {
        "examples": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": mean(record["correct"] for record in records),
        "mean_tokens": mean(token_counts),
        "total_tokens": total_tokens,
        "capped": sum(
            record["finish_reason"] == "length" and record["tokens"] >= max_tokens
            for record in records
        ),
        "boundary_tokens": sum(record["boundary_tokens"] for record in records),
        "generation_seconds": seconds,
        "tokens_per_second": total_tokens / seconds if seconds else None,
        "grading_errors": sum(record["grading_error"] is not None for record in records),
    }


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
    for example, request_output in zip(examples, outputs, strict=True):
        output = request_output.outputs[0]
        token_ids = list(output.token_ids)
        correct, grading_error = grade_answer(example["answer"], output.text)
        records.append(
            {
                "unique_id": example["unique_id"],
                "subject": example.get("subject"),
                "level": example.get("level"),
                "gold": example["answer"],
                "correct": correct,
                "grading_error": grading_error,
                "tokens": len(token_ids),
                "boundary_tokens": sum(token_id in boundary_ids for token_id in token_ids),
                "finish_reason": output.finish_reason,
                "text": output.text,
            }
        )
    return records, seconds


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    dataset_path = Path(args.dataset).resolve()
    vector_dir = Path(args.vector_dir).resolve()
    output_path = Path(args.output).resolve()

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("--max-model-len must be greater than --max-tokens")

    ensure_dataset(dataset_path)
    examples = load_examples(dataset_path, args.offset, args.limit)
    vector_paths = {
        name: require_file(vector_dir / f"{name}_avg_vector.gguf")
        for name in ("execution", "reflection", "transition")
    }

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    newline_ids = sorted(
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token.endswith("ĊĊ")
    )
    if not newline_ids:
        raise RuntimeError("No paragraph-break token ids ending in 'ĊĊ' found")

    steering = SteeringSpec(
        conflict="sequential",
        vectors=[
            VectorSpec(
                name="execution",
                source=vector_paths["execution"],
                scale=args.scale,
                layers=[args.layer],
                apply=ApplySpec(generation_tokens=newline_ids),
            ),
            VectorSpec(
                name="reflection",
                source=vector_paths["reflection"],
                scale=-args.scale,
                layers=[args.layer],
                apply=ApplySpec(generation_tokens=newline_ids),
            ),
            VectorSpec(
                name="transition",
                source=vector_paths["transition"],
                scale=-args.scale,
                layers=[args.layer],
                apply=ApplySpec(generation_tokens=newline_ids),
            ),
        ],
    )
    print(f"Preflight: {len(examples)} examples, {len(newline_ids)} boundary ids")

    result: dict[str, Any] = {
        "scope": "EasySteer_SEAL_MATH500_paired_evaluation",
        "protocol": {
            "model": str(model_path),
            "dataset": str(dataset_path),
            "dataset_url": DATASET_URL,
            "offset": args.offset,
            "limit": args.limit,
            "layer": args.layer,
            "scale": args.scale,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "temperature": 0.0,
            "boundary_token_id_count": len(newline_ids),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
            steer_multi_vector=True,
            enforce_eager=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            seed=args.seed,
        )
        result["startup_seconds"] = time.perf_counter() - started
        result["environment"]["gpu"] = torch.cuda.get_device_name(0)

        prompts = [build_prompt(example["problem"]) for example in examples]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            skip_special_tokens=False,
        )
        boundary_set = set(newline_ids)

        print("Running baseline...")
        baseline, baseline_seconds = generate_records(
            llm, prompts, examples, params, boundary_set, steering=None
        )
        result["baseline"] = {
            "summary": summarize(baseline, baseline_seconds, args.max_tokens),
            "records": baseline,
        }
        write_result(output_path, result)
        print(f"Baseline checkpoint: {output_path}")

        print("Running SEAL...")
        seal, seal_seconds = generate_records(
            llm, prompts, examples, params, boundary_set, steering=steering
        )
        result["seal"] = {
            "summary": summarize(seal, seal_seconds, args.max_tokens),
            "records": seal,
        }

        improved = [
            base["unique_id"]
            for base, steered in zip(baseline, seal, strict=True)
            if not base["correct"] and steered["correct"]
        ]
        degraded = [
            base["unique_id"]
            for base, steered in zip(baseline, seal, strict=True)
            if base["correct"] and not steered["correct"]
        ]
        baseline_summary = result["baseline"]["summary"]
        seal_summary = result["seal"]["summary"]
        result["comparison"] = {
            "mean_token_change": seal_summary["mean_tokens"] - baseline_summary["mean_tokens"],
            "mean_token_change_percent": (
                seal_summary["mean_tokens"] / baseline_summary["mean_tokens"] - 1
            )
            * 100,
            "accuracy_change_percentage_points": (
                seal_summary["accuracy"] - baseline_summary["accuracy"]
            )
            * 100,
            "improved": improved,
            "degraded": degraded,
        }
        write_result(output_path, result)

        print("\n===== PAIRED SUMMARY =====")
        print(f"Examples: {args.limit} (offset {args.offset})")
        print(
            f"Baseline: acc={baseline_summary['accuracy']:.3f}, "
            f"mean_tokens={baseline_summary['mean_tokens']:.1f}, "
            f"capped={baseline_summary['capped']}, "
            f"time={baseline_summary['generation_seconds']:.1f}s"
        )
        print(
            f"SEAL:     acc={seal_summary['accuracy']:.3f}, "
            f"mean_tokens={seal_summary['mean_tokens']:.1f}, "
            f"capped={seal_summary['capped']}, "
            f"time={seal_summary['generation_seconds']:.1f}s"
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
