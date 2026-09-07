# -*- coding: utf-8 -*-
"""
Steered code-generation inference and evaluation.

This script runs sampled code generation with a Qwen2 model that supports
activation steering, evaluates generated solutions with the project's code
evaluation utilities, and stores both strict and relaxed evaluation metrics.

Main features
-------------
1. Multi-GPU data parallel inference
   - Each GPU processes a disjoint subset of benchmark examples.
   - Each worker writes an append-only shard file.
   - Existing shard rows are skipped, so interrupted runs can be resumed.

2. Activation steering
   - A precomputed steering vector is loaded from disk.
   - The vector is injected into the configured transformer layer through the
     custom ``Qwen2ForCausalLM`` implementation.

3. Code benchmark evaluation
   - Strict pass@1 is computed by ``codegen_metrics``.
   - Relaxed metrics are computed from per-instance execution metadata.
   - Token and length statistics are collected for downstream analysis.

Expected dataset naming
-----------------------
The script targets code-generation benchmarks whose CLI dataset name starts
with ``Code_``. The benchmark file is expected at:

    <dataset_dir>/<dataset>/test.jsonl

The release name passed to ``load_code_generation_dataset`` is the part after
the first underscore. For example:

    --dataset Code_livecodebenchv2

will call:

    load_code_generation_dataset(
        release_version="livecodebenchv2",
        local_path="<dataset_dir>/Code_livecodebenchv2/test.jsonl",
    )

Outputs
-------
For a model basename ``Qwen2.5-7B-Instruct`` and dataset ``Code_xxx``, files are
written under:

    <output_path>/Qwen2.5-7B-Instruct/Code_xxx/

The base filename is:

    [<run_id>_]steer_temp<T>_maxlen<N>

Generated files include:

    *.shard<R>.jsonl       Per-rank raw generations.
    *.predictions.jsonl    Benchmark instances with inserted generations.
    *.metrics.json         Strict, relaxed, and length/token metrics.
    *.code_eval.jsonl      Per-instance evaluation details.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# The custom model must provide:
#   - set_steering_flag(...)
#   - start_new_round()
# and should implement the actual layer-wise steering logic internally.
from modeling_utils.modeling_qwen2_dynamic_3D import Qwen2ForCausalLM

# Project-specific code-evaluation utilities.
from code_evaluation import (
    codegen_metrics,
    extract_code,
    extract_instance_results,
    get_deepseekcode_question_template_answer,
    load_code_generation_dataset,
)


# ---------------------------------------------------------------------------
# Model-specific dynamic steering defaults
# ---------------------------------------------------------------------------

# These values are used only when the corresponding --dyn_* argument is not
# supplied explicitly. Matching is substring-based against --model_name_or_path,
# so both local paths and Hugging Face model IDs are supported.
MODEL_DYN_DEFAULTS: Dict[str, Dict[str, float]] = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "dyn_q25c": 0.662293,
        "dyn_q75c": 0.94805,
        "dyn_low_val_1": -1.02,
        "dyn_q25v": 0.000560,
        "dyn_q75v": 0.011597,
        "dyn_low_val_2": -1.91,
        "dyn_high_val_2": 0.1,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "dyn_q25c": 0.666017,
        "dyn_q75c": 0.927745,
        "dyn_low_val_1": -1.19,
        "dyn_q25v": 0.000488,
        "dyn_q75v": 0.009931,
        "dyn_low_val_2": -2.34,
        "dyn_high_val_2": 0.1,
    },
    "QwQ-32B": {
        "dyn_q25c": 0.700670,
        "dyn_q75c": 0.917506,
        "dyn_low_val_1": -1.31,
        "dyn_q25v": 0.000386,
        "dyn_q75v": 0.007279,
        "dyn_low_val_2": -2.73,
        "dyn_high_val_2": 0.1,
    },
}

DYN_HPARAM_NAMES: Tuple[str, ...] = (
    "dyn_q25c",
    "dyn_q75c",
    "dyn_low_val_1",
    "dyn_q25v",
    "dyn_q75v",
    "dyn_low_val_2",
    "dyn_high_val_2",
)


def apply_model_dynamic_defaults(args: argparse.Namespace) -> Optional[str]:
    """Fill missing dynamic steering hyperparameters from model defaults.

    User-specified CLI values always take precedence. The function returns the
    matched model key, or ``None`` when no default profile matches.
    """

    matched_model: Optional[str] = None

    for model_key, defaults in MODEL_DYN_DEFAULTS.items():
        if model_key in args.model_name_or_path:
            matched_model = model_key
            for name in DYN_HPARAM_NAMES:
                if getattr(args, name) is None:
                    setattr(args, name, float(defaults[name]))
            break

    return matched_model


def build_dynamic_hparams(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    """Build the ``dyn_hparams`` payload expected by the custom model."""

    return {
        "q25c": args.dyn_q25c,
        "q75c": args.dyn_q75c,
        "low_val_1": args.dyn_low_val_1,
        "q25v": args.dyn_q25v,
        "q75v": args.dyn_q75v,
        "low_val_2": args.dyn_low_val_2,
        "high_val_2": args.dyn_high_val_2,
    }


def print_dynamic_hparams(args: argparse.Namespace, matched_model: Optional[str]) -> None:
    """Print the dynamic steering hyperparameters used for this run."""

    if matched_model is None:
        print("[main] dynamic steering defaults: no matched model profile")
    else:
        print(f"[main] dynamic steering defaults: matched {matched_model}")

    print("[main] dynamic steering hyperparameters:")
    for name in DYN_HPARAM_NAMES:
        print(f"[main]   {name} = {getattr(args, name)}")


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

def configure_quiet_runtime() -> None:
    """Reduce noisy warnings and logs from common ML dependencies.

    This function only changes logging verbosity and warning display behavior.
    It does not change model outputs, sampling behavior, or evaluation results.
    """

    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings("ignore")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        # Logging setup must never prevent inference from starting.
        pass

    try:
        import datasets

        datasets.utils.logging.set_verbosity_error()
    except Exception:
        pass

    try:
        import numpy as np

        np.seterr(all="ignore")
    except Exception:
        pass


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible sampling.

    The generation path uses explicit token-by-token top-p sampling, matching
    ``transformer_inference_steer_dp.py``.
    """

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These flags prefer deterministic kernels when available. Some CUDA
    # operations may still have hardware- or version-dependent behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Path helpers and resumability
# ---------------------------------------------------------------------------

def build_output_paths(args: argparse.Namespace) -> Tuple[Path, str]:
    """Return the output directory and run-specific filename prefix.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    output_dir:
        Directory where all generated artifacts for this run are stored.
    base_name:
        Common filename prefix shared by shard, prediction, metric, and eval
        files.
    """

    model_basename = Path(args.model_name_or_path).expanduser().resolve().name
    output_dir = Path(args.output_path) / model_basename / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    run_prefix = f"{args.run_id.strip()}_" if args.run_id else ""
    base_name = f"{run_prefix}steer_temp{args.temperature}_maxlen{args.max_generated_tokens}"

    return output_dir, base_name


def load_existing_indices(shard_file: Path) -> Set[int]:
    """Load already completed example indices from a shard file.

    The worker writes one JSON object per line with at least an ``idx`` field.
    When a run is restarted, those indices are skipped so that the script can
    resume without overwriting previous generations.

    Malformed lines are ignored instead of failing the whole run, because shard
    files can be interrupted during appends.
    """

    existing_indices: Set[int] = set()

    if not shard_file.exists():
        return existing_indices

    with shard_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if "idx" in obj:
                    existing_indices.add(int(obj["idx"]))
            except Exception:
                continue

    return existing_indices


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to a UTF-8 JSONL file."""

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def top_p_sampling_step(
    last_logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, float]:
    """Sample one token with the same top-p strategy as the math inference script."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    logits = last_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumsum > top_p
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_sorted_idx)

    chosen_prob = sorted_probs.gather(-1, next_sorted_idx)
    next_logprob = torch.log(chosen_prob + 1e-12).item()

    return next_token, next_logprob


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------

def setup_attention_backend(model: torch.nn.Module) -> None:
    """Prefer FlashAttention 2 when available; otherwise use PyTorch SDPA.

    The custom Qwen2 model follows the Hugging Face configuration convention.
    If ``model.config.attn_implementation`` exists, this function sets it to
    either ``flash_attention_2`` or ``sdpa``.

    Failure to configure a specific backend is non-fatal; the model will fall
    back to its default attention implementation.
    """

    try:
        use_flash_attention_2 = False

        try:
            from transformers.utils import is_flash_attn_2_available

            use_flash_attention_2 = bool(is_flash_attn_2_available())
            if use_flash_attention_2:
                # Importing this module eagerly catches broken installations
                # before generation starts.
                import flash_attn_2_cuda  # noqa: F401
        except Exception:
            use_flash_attention_2 = False

        if not (hasattr(model, "config") and hasattr(model.config, "attn_implementation")):
            return

        if use_flash_attention_2:
            model.config.attn_implementation = "flash_attention_2"
            print("[INFO] attention backend: flash_attention_2")
        else:
            model.config.attn_implementation = "sdpa"

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                )

            print("[INFO] attention backend: PyTorch SDPA")

    except Exception as exc:
        print(f"[WARN] attention backend setup failed: {exc}")


def load_model_and_tokenizer(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase, torch.dtype]:
    """Load tokenizer and steered Qwen2 model on the requested device."""

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    # Left padding requires a valid pad token. Most instruction-tuned Qwen
    # tokenizers can safely use EOS as PAD for inference.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    setup_attention_backend(model)
    model.eval()

    return model, tokenizer, dtype


def load_steering_vector(
    steer_vector_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load and cast the steering vector used by the custom model."""

    steer_vector = torch.load(steer_vector_path, map_location="cpu")

    if not isinstance(steer_vector, torch.Tensor):
        raise TypeError(
            f"Expected steering vector to be a torch.Tensor, got {type(steer_vector)!r}."
        )

    return steer_vector.to(device=device, dtype=dtype)


def enable_model_steering(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    steer_vector: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Configure activation steering in the custom Qwen2 model."""

    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
        dyn_hparams=build_dynamic_hparams(args),
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def is_code_dataset(dataset_name: str) -> bool:
    """Return whether a dataset name follows this script's code benchmark convention."""

    return dataset_name.lower().startswith("code_")


def dataset_to_release(dataset_name: str) -> str:
    """Convert a CLI dataset name into a benchmark release name."""

    return dataset_name.split("_", 1)[1] if "_" in dataset_name else dataset_name


def prepare_code_benchmark(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[List[Any], List[str]]:
    """Load benchmark instances and render chat prompts.

    The benchmark loader returns project-specific instance objects. Each object
    is converted into a prompt with ``get_deepseekcode_question_template_answer``,
    then wrapped by the tokenizer's chat template.
    """

    release = dataset_to_release(args.dataset)
    dataset_path = Path(args.dataset_dir) / args.dataset / "test.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Code dataset file not found. Expected "
            f"--dataset_dir/--dataset/test.jsonl, got: {dataset_path}"
        )

    benchmark = load_code_generation_dataset(
        release_version=release,
        local_path=str(dataset_path),
    )

    prompts: List[str] = []

    for instance in benchmark:
        user_prompt = get_deepseekcode_question_template_answer(instance)
        messages = [{"role": "user", "content": user_prompt}]

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(chat_prompt)

    return benchmark, prompts


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_worker_indices(num_examples: int, rank: int, world_size: int) -> List[int]:
    """Assign examples to a worker by modulo sharding."""

    return [idx for idx in range(num_examples) if idx % world_size == rank]


@torch.no_grad()
def sample_with_tracking(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[float]]:
    """Sample tokens while tracking per-step log probabilities."""

    device = input_ids.device
    token_logprobs: List[float] = []
    generated: List[int] = []

    past_key_values = None

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=(dtype in (torch.float16, torch.bfloat16)),
        dtype=dtype,
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=None,
        )
        past_key_values = outputs.past_key_values

    last_token = None
    for _ in range(max_new_tokens):
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=(dtype in (torch.float16, torch.bfloat16)),
            dtype=dtype,
        ):
            if last_token is None:
                logits = outputs.logits[:, -1, :]
            else:
                out = model(
                    input_ids=last_token,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

        next_token, next_logprob = top_p_sampling_step(logits, temperature, top_p)
        token_logprobs.append(next_logprob)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            generated.append(next_token.item())
            break

        generated.append(next_token.item())
        last_token = next_token

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), []

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs


def generate_one(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dtype: torch.dtype,
) -> str:
    """Generate one model response using the shared top-p sampling path."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids, _ = sample_with_tracking(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            dtype=dtype,
        )

    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    del inputs, gen_ids

    return decoded


def cleanup_cuda_memory() -> None:
    """Release unused Python and CUDA memory after each example."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Run one inference worker.

    Each worker:
    1. Selects its CUDA device.
    2. Loads the model, tokenizer, and steering vector.
    3. Processes only examples where ``idx % world_size == rank``.
    4. Appends generation records to its rank-specific shard file.
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    output_dir, base_name = build_output_paths(args)
    shard_file = output_dir / f"{base_name}.shard{rank}.jsonl"

    existing_indices = load_existing_indices(shard_file)

    model, tokenizer, dtype = load_model_and_tokenizer(args, device)
    steer_vector = load_steering_vector(args.steer_vector_path, device, dtype)
    enable_model_steering(model, tokenizer, steer_vector, args)

    if not is_code_dataset(args.dataset):
        raise ValueError(
            "This script only supports code-generation datasets whose name starts "
            f"with 'Code_'. Got: {args.dataset!r}."
        )

    _, prompts = prepare_code_benchmark(args, tokenizer)

    my_indices = get_worker_indices(
        num_examples=len(prompts),
        rank=rank,
        world_size=world_size,
    )

    progress = tqdm(
        total=len(my_indices),
        desc=f"Rank {rank} code inference",
        position=rank,
        leave=True,
    )

    for idx in my_indices:
        if idx in existing_indices:
            progress.update(1)
            continue

        # Reset per-example steering state if the custom model tracks generation
        # rounds internally.
        model.start_new_round()

        prompt = prompts[idx]

        try:
            output_text = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_generated_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM][rank {rank}] skipped idx={idx}")
            cleanup_cuda_memory()
            progress.update(1)
            continue
        except Exception as exc:
            print(f"[ERROR][rank {rank}] skipped idx={idx}: {exc}")
            cleanup_cuda_memory()
            progress.update(1)
            continue

        append_jsonl(
            shard_file,
            {
                "idx": idx,
                "prompt": prompt,
                "output_text": output_text,
            },
        )

        cleanup_cuda_memory()
        progress.update(1)

    progress.close()
    print(f"[rank {rank}] done. Shard saved to: {shard_file}")


# ---------------------------------------------------------------------------
# Aggregation and evaluation helpers
# ---------------------------------------------------------------------------

def read_shard_outputs(shard_files: Sequence[Path]) -> Dict[int, str]:
    """Read all shard files and keep the latest output per index."""

    idx_to_text: Dict[int, str] = {}

    for shard_file in shard_files:
        with shard_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    idx_to_text[int(obj["idx"])] = obj["output_text"]
                except Exception:
                    continue

    return idx_to_text


def collect_existing_shards(output_dir: Path, base_name: str, num_gpus: int) -> List[Path]:
    """Return shard paths that currently exist for this run."""

    return [
        output_dir / f"{base_name}.shard{rank}.jsonl"
        for rank in range(max(1, num_gpus))
        if (output_dir / f"{base_name}.shard{rank}.jsonl").exists()
    ]


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write a JSON file with UTF-8 encoding."""

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def extract_generated_text(generated: Any) -> str:
    """Best-effort extraction of text from common generation containers."""

    if isinstance(generated, str):
        return generated

    if isinstance(generated, Mapping):
        for key in (
            "text",
            "content",
            "generated_response",
            "generated_text",
            "output",
            "output_text",
            "message",
            "response",
        ):
            value = generated.get(key)
            if isinstance(value, str):
                return value

    if isinstance(generated, list):
        for item in generated:
            text = extract_generated_text(item)
            if text:
                return text

    return ""


def ratio_passes(grade_list: Any, threshold: float) -> Tuple[bool, int, int]:
    """Return whether a test-pass ratio meets a threshold."""

    if isinstance(grade_list, (list, tuple)) and len(grade_list) > 0:
        total = len(grade_list)
        passed = sum(bool(item) for item in grade_list)
        return (passed / total) >= threshold, passed, total

    return False, 0, 0


def metadata_indicates_compile_or_output(metadata: Any) -> bool:
    """Infer a weak success signal from execution metadata.

    This metric is intentionally heuristic. It is useful for exploratory
    analysis but should not replace strict pass@1 when reporting benchmark
    performance.
    """

    if not isinstance(metadata, Mapping):
        return False

    compile_keys = (
        "compile",
        "compile_success",
        "compiled",
        "build_success",
        "compilation_success",
    )
    for key in compile_keys:
        if key in metadata and bool(metadata[key]):
            return True

    stdout_keys = (
        "stdout",
        "output",
        "std_output",
        "run_stdout",
    )
    for key in stdout_keys:
        value = metadata.get(key)

        if isinstance(value, str) and value.strip():
            return True

        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True

    if metadata.get("num_total"):
        num_passed = float(metadata.get("num_passed", 0))
        num_total = max(1.0, float(metadata["num_total"]))
        return (num_passed / num_total) > 0

    return False


def metadata_indicates_timeout(metadata: Any) -> bool:
    """Return whether metadata marks the execution as timed out."""

    if not isinstance(metadata, Mapping):
        return False

    timeout_keys = (
        "timeout",
        "timed_out",
        "time_limit_exceeded",
    )
    return any(bool(metadata.get(key)) for key in timeout_keys)


def mean_boolean(flags: Sequence[bool]) -> float:
    """Compute the fraction of truthy values in a sequence."""

    if not flags:
        return 0.0

    return sum(1 for item in flags if item) / len(flags)


def compute_relaxed_metrics(
    graded: Sequence[Any],
    metadatas: Sequence[Any],
    relaxed_min_ratio: float,
    relaxed_timeout_as_pass: bool,
) -> Dict[str, Any]:
    """Compute per-instance relaxed success rates."""

    any_test_flags: List[bool] = []
    ratio_flags: List[bool] = []
    compile_or_output_flags: List[bool] = []
    timeout_flags: List[bool] = []
    combined_relaxed_flags: List[bool] = []

    for grade_list, metadata in zip(graded, metadatas):
        any_test = bool(grade_list) and isinstance(grade_list, (list, tuple)) and any(grade_list)
        ratio_ok, _, _ = ratio_passes(grade_list, relaxed_min_ratio)
        compile_or_output_ok = metadata_indicates_compile_or_output(metadata)
        timeout = metadata_indicates_timeout(metadata)

        combined_ok = (
            any_test
            or ratio_ok
            or compile_or_output_ok
            or (timeout and relaxed_timeout_as_pass)
        )

        any_test_flags.append(any_test)
        ratio_flags.append(ratio_ok)
        compile_or_output_flags.append(compile_or_output_ok)
        timeout_flags.append(timeout)
        combined_relaxed_flags.append(combined_ok)

    return {
        "any_test_rate": mean_boolean(any_test_flags),
        "ratio_thr": relaxed_min_ratio,
        "ratio_rate": mean_boolean(ratio_flags),
        "compile_or_output_rate": mean_boolean(compile_or_output_flags),
        "combined_relaxed_rate": mean_boolean(combined_relaxed_flags),
        "num_timeout": sum(1 for item in timeout_flags if item),
        "timeout_as_pass": relaxed_timeout_as_pass,
    }


def split_think_text(text: str) -> Tuple[str, bool]:
    """Return text before a closing think tag and whether such a tag was found.

    The function supports both literal ``</think>`` and HTML-escaped
    ``&lt;/think&gt;`` tags.
    """

    lowered = text.lower()
    tag_idx = lowered.find("</think>")

    if tag_idx == -1:
        tag_idx = lowered.find("&lt;/think&gt;")

    if tag_idx == -1:
        return text, False

    return text[:tag_idx], True


def compute_length_stats(
    outputs: Sequence[Sequence[Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Compute word, token, and think-block token statistics."""

    response_word_lengths: List[int] = []
    response_token_lengths: List[int] = []
    think_token_lengths: List[int] = []

    think_found = 0
    fallback_full = 0

    for output_list in outputs:
        text = extract_generated_text(output_list[0]) if output_list else ""

        response_word_lengths.append(len(text.split()) if text else 0)

        response_token_lengths.append(
            len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
        )

        think_text, found = split_think_text(text)
        if found:
            think_found += 1
        elif text:
            fallback_full += 1

        think_token_lengths.append(
            len(tokenizer(think_text, add_special_tokens=False)["input_ids"]) if think_text else 0
        )

    num_samples = len(outputs)

    return {
        "avg_words": (sum(response_word_lengths) / num_samples) if num_samples else 0.0,
        "avg_tokens": (sum(response_token_lengths) / num_samples) if num_samples else 0.0,
        "avg_think_tokens": (sum(think_token_lengths) / num_samples) if num_samples else 0.0,
        "think_found": think_found,
        "fallback_full": fallback_full,
        "num_samples": num_samples,
    }


# ---------------------------------------------------------------------------
# Aggregation and evaluation
# ---------------------------------------------------------------------------

def aggregate_and_evaluate(args: argparse.Namespace) -> None:
    """Merge shard outputs, run code evaluation, and save metrics."""

    print("[main] aggregating shards and running evaluation")

    output_dir, base_name = build_output_paths(args)
    shard_files = collect_existing_shards(output_dir, base_name, args.num_gpus)

    if not shard_files:
        raise FileNotFoundError(
            f"No shard files found in {output_dir} for base name {base_name!r}."
        )

    idx_to_text = read_shard_outputs(shard_files)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    benchmark, _ = prepare_code_benchmark(args, tokenizer)

    num_examples = len(benchmark)
    missing_indices = sorted(set(range(num_examples)) - set(idx_to_text.keys()))

    if missing_indices:
        print(
            f"[main][WARN] missing {len(missing_indices)} / {num_examples} examples; "
            "evaluation will use only completed generations"
        )

    paired_outputs = sorted(
        ((idx, idx_to_text[idx]) for idx in idx_to_text.keys()),
        key=lambda item: item[0],
    )

    outputs = [[text] for _, text in paired_outputs]
    extracted = [[extract_code(text) for text in output_list] for output_list in outputs]
    sub_benchmark = [benchmark[idx] for idx, _ in paired_outputs]

    save_results = [
        instance.insert_output(output_list, extracted_list)
        for instance, output_list, extracted_list in zip(sub_benchmark, outputs, extracted)
    ]

    prediction_path = output_dir / f"{base_name}.predictions.jsonl"
    save_json(prediction_path, save_results)

    eval_samples = [instance.get_evaluation_sample() for instance in sub_benchmark]
    generations = extracted

    timeout = max(1, int(args.eval_timeout))

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=args.num_process_evaluate,
        timeout=timeout,
    )

    strict_summary = metrics[0]
    pass_at_1 = strict_summary.get("pass@1")

    print(f"[main] pass@1 = {pass_at_1} (timeout={timeout}s)")

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]

    relaxed_summary = compute_relaxed_metrics(
        graded=graded,
        metadatas=metadatas,
        relaxed_min_ratio=args.relaxed_min_ratio,
        relaxed_timeout_as_pass=args.relaxed_timeout_as_pass,
    )
    relaxed_summary["timeout_sec"] = timeout

    print(f"[main] relaxed_any_test_rate     = {relaxed_summary['any_test_rate']:.4f}")
    print(
        f"[main] relaxed_ratio>={args.relaxed_min_ratio:.2f}     = "
        f"{relaxed_summary['ratio_rate']:.4f}"
    )
    print(
        f"[main] relaxed_compile_or_output = "
        f"{relaxed_summary['compile_or_output_rate']:.4f}"
    )
    print(
        f"[main] combined_relaxed_rate     = "
        f"{relaxed_summary['combined_relaxed_rate']:.4f} "
        f"(timeout_as_pass={args.relaxed_timeout_as_pass})"
    )

    save_eval_results = [
        instance.insert_output_evaluation(
            output_list,
            extracted_list,
            grade_list,
            metadata=metadata,
        )
        for instance, output_list, extracted_list, grade_list, metadata in zip(
            sub_benchmark,
            outputs,
            extracted,
            graded,
            metadatas,
        )
    ]

    length_stats = compute_length_stats(outputs, tokenizer)

    print(f"[main] avg words          = {length_stats['avg_words']}")
    print(f"[main] avg tokens         = {length_stats['avg_tokens']}")
    print(f"[main] avg think tokens   = {length_stats['avg_think_tokens']}")
    print(
        f"[main] think blocks found = "
        f"{length_stats['think_found']}/{length_stats['num_samples']} "
        f"(fallback_full={length_stats['fallback_full']})"
    )

    metrics_out = {
        "strict": strict_summary,
        "relaxed": relaxed_summary,
        "length_stats": length_stats,
    }

    metrics_path = output_dir / f"{base_name}.metrics.json"
    code_eval_path = output_dir / f"{base_name}.code_eval.jsonl"

    save_json(metrics_path, metrics_out)
    save_json(code_eval_path, save_eval_results)

    print(f"[main] saved predictions to: {prediction_path}")
    print(f"[main] saved metrics to: {metrics_path}")
    print(f"[main] saved per-instance eval to: {code_eval_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run activation-steered sampled code generation and evaluate "
            "the resulting programs."
        )
    )

    # General I/O and execution settings.
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--run_id", type=str, default="")

    # Generation settings.
    parser.add_argument("--max_generated_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Steering settings.
    parser.add_argument("--steer_vector_path", type=str, required=True)
    parser.add_argument("--steer_layer", type=int, default=22)
    parser.add_argument("--steer_coef", type=float, default=1.0)

    # Dynamic steering hyperparameters. Leave them unspecified to use the
    # model-specific defaults in MODEL_DYN_DEFAULTS when available. Passing any
    # value on the command line overrides only that value.
    parser.add_argument("--dyn_q25c", type=float, default=None)
    parser.add_argument("--dyn_q75c", type=float, default=None)
    parser.add_argument("--dyn_low_val_1", type=float, default=None)
    parser.add_argument("--dyn_q25v", type=float, default=None)
    parser.add_argument("--dyn_q75v", type=float, default=None)
    parser.add_argument("--dyn_low_val_2", type=float, default=None)
    parser.add_argument("--dyn_high_val_2", type=float, default=None)

    # Evaluation settings.
    parser.add_argument(
        "--eval_timeout",
        type=int,
        default=180,
        help="Timeout in seconds for code execution during strict evaluation.",
    )
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of parallel processes used by the code evaluator.",
    )
    parser.add_argument(
        "--relaxed_min_ratio",
        type=float,
        default=0.25,
        help="Relaxed pass if passed_tests / total_tests is at least this value.",
    )
    parser.add_argument(
        "--relaxed_timeout_as_pass",
        action="store_true",
        default=True,
        help="Count timeout cases as pass in the combined relaxed metric.",
    )
    parser.add_argument(
        "--strict_timeout_as_fail",
        dest="relaxed_timeout_as_pass",
        action="store_false",
        help="Do not count timeout cases as pass in the combined relaxed metric.",
    )

    # Kept for backward CLI compatibility. The current implementation folds
    # these signals into the compile_or_output relaxed heuristic.
    parser.add_argument("--relaxed_compile_as_pass", action="store_true", default=True)
    parser.add_argument("--relaxed_nonempty_output_as_pass", action="store_true", default=True)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before launching workers."""

    if args.num_gpus < 1:
        raise ValueError(f"--num_gpus must be >= 1, got {args.num_gpus}.")

    if args.max_generated_tokens < 1:
        raise ValueError(
            f"--max_generated_tokens must be >= 1, got {args.max_generated_tokens}."
        )

    if not 0.0 <= args.relaxed_min_ratio <= 1.0:
        raise ValueError(
            f"--relaxed_min_ratio must be in [0, 1], got {args.relaxed_min_ratio}."
        )

    if not is_code_dataset(args.dataset):
        raise ValueError(
            "This script only supports code-generation datasets whose name starts "
            f"with 'Code_'. Got: {args.dataset!r}."
        )


def main() -> None:
    """Entry point."""

    configure_quiet_runtime()

    parser = build_arg_parser()
    args = parser.parse_args()

    matched_model = apply_model_dynamic_defaults(args)
    print_dynamic_hparams(args, matched_model)

    validate_args(args)
    set_seed(42)

    world_size = max(1, int(args.num_gpus))

    if world_size == 1:
        worker(rank=0, world_size=1, args=args)
    else:
        mp.spawn(
            worker,
            nprocs=world_size,
            args=(world_size, args),
        )

    aggregate_and_evaluate(args)


if __name__ == "__main__":
    main()
