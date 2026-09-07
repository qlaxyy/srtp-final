"""Small, deterministic SEAL steering smoke test for EasySteer v2.

This checks that the bundled execution/reflection/transition vectors can be
loaded and applied at paragraph-break tokens.  It is a mechanism test, not a
paper-level effectiveness evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec


DEFAULT_MODEL = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OUTPUT = "/root/autodl-tmp/results/easysteer/seal_smoke.json"
DEFAULT_PROBLEM = (
    "A store initially has 120 notebooks. It sells 25% of them in the "
    "morning and then sells one third of the remaining notebooks in the "
    "afternoon. How many notebooks remain? Reason step by step, separate "
    "reasoning steps with blank lines, and put the final answer within "
    "\\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--vector-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    vector_dir = Path(args.vector_dir).resolve()

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")

    vector_paths = {
        "execution": require_file(vector_dir / "execution_avg_vector.gguf"),
        "reflection": require_file(vector_dir / "reflection_avg_vector.gguf"),
        "transition": require_file(vector_dir / "transition_avg_vector.gguf"),
    }

    # Validate the current v2 selector schema before paying the engine startup
    # cost.  A bare ``tokens=`` selector belonged to an older API.
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
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
    print(f"SteeringSpec validation: OK ({len(newline_ids)} boundary ids)")

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
        startup_seconds = time.perf_counter() - started
        print(f"Engine startup: {startup_seconds:.2f}s")

        prompt = (
            "Please reason step by step, and put your final answer within "
            f"\\boxed{{}}.\nUser: {args.problem}\nAssistant: <think>"
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            skip_special_tokens=False,
        )

        started = time.perf_counter()
        baseline = llm.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0].outputs[0]
        baseline_seconds = time.perf_counter() - started

        started = time.perf_counter()
        steered = llm.generate(
            [prompt],
            sampling_params=sampling_params,
            steering=steering,
            use_tqdm=False,
        )[0].outputs[0]
        steered_seconds = time.perf_counter() - started

        boundary_set = set(newline_ids)
        baseline_ids = list(baseline.token_ids)
        steered_ids = list(steered.token_ids)
        result = {
            "scope": "mechanism_smoke_test_not_paper_evaluation",
            "model": str(model_path),
            "vector_dir": str(vector_dir),
            "layer": args.layer,
            "scale": args.scale,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "boundary_token_id_count": len(newline_ids),
            "startup_seconds": startup_seconds,
            "baseline": {
                "tokens": len(baseline_ids),
                "boundary_tokens": sum(t in boundary_set for t in baseline_ids),
                "seconds": baseline_seconds,
                "finish_reason": baseline.finish_reason,
                "text": baseline.text,
            },
            "seal": {
                "tokens": len(steered_ids),
                "boundary_tokens": sum(t in boundary_set for t in steered_ids),
                "seconds": steered_seconds,
                "finish_reason": steered.finish_reason,
                "text": steered.text,
            },
            "token_ids_identical": baseline_ids == steered_ids,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n===== SUMMARY =====")
        print("Baseline tokens:", result["baseline"]["tokens"])
        print("SEAL tokens:", result["seal"]["tokens"])
        print(
            "Token change:",
            result["seal"]["tokens"] - result["baseline"]["tokens"],
        )
        print("Baseline boundaries:", result["baseline"]["boundary_tokens"])
        print("SEAL boundaries:", result["seal"]["boundary_tokens"])
        print(f"Baseline time: {baseline_seconds:.2f}s")
        print(f"SEAL time: {steered_seconds:.2f}s")
        print("Token IDs identical:", result["token_ids_identical"])
        print("Result file:", output_path)
        print("\n===== BASELINE =====")
        print(baseline.text)
        print("\n===== SEAL =====")
        print(steered.text)
    finally:
        del llm
        gc.collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
