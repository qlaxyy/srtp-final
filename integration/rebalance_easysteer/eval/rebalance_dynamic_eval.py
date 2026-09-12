"""Paired GSM8K evaluation of author-code ReBalance on EasySteer vLLM."""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import version as package_version
import inspect
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
from runtime_guards import guard_dynamic_preemption
from decode_profiler import ProfileWindowComplete
from calibration_contract import resolve_calibration_layer
from baseline_reuse import execution_identity, validate_saved_baseline

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
    parser.add_argument("--layer", type=int,
                        help="Use calibration placement when available; otherwise decoder18")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--async-scheduling", action="store_true",
                        help="Reuse the verified 1.5B runtime scheduling option")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Diagnostic CPU/CUDA trace only; timings include overhead")
    parser.add_argument("--profile-start-step", type=int, default=16)
    parser.add_argument("--profile-only", action="store_true",
                        help="Stop after the trace window; requires --diagnostic-group")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-timeout-seconds", type=int, default=1500)
    parser.add_argument("--resume-result", type=Path, help="Continue a matching interrupted pair into a NEW output")
    parser.add_argument("--resume-elapsed-seconds", type=float, default=0,
                        help="Prior interrupted arm wall time; retained in total cost")
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
    parser.add_argument("--feedback-config", type=Path,
                        help="Fixed readout metadata for negative-displacement clipping")
    parser.add_argument("--feedback-disabled", action="store_true",
                        help="Engineering equivalence check with feedback payload disabled")
    parser.add_argument("--radial-restore", choices=["off", "on"],
                        help="Same-graph norm restoration control or candidate")
    parser.add_argument("--negative-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Ablate positive coefficients after the unchanged dynamic rule")
    parser.add_argument("--sampled-confidence", action="store_true",
                        help="Use raw probability of the actually sampled token")
    parser.add_argument("--baseline-result", type=Path,
                        help="Reuse a complete control with verified code/model content identity")
    parser.add_argument("--dynamic-first", action="store_true",
                        help="Check the intervention arm before spending time on baseline")
    parser.add_argument("--diagnostic-group", choices=["baseline", "rebalance_dynamic"],
                        help="Engineering-only single arm; never a formal method comparison")
    args = parser.parse_args()
    if args.sampled_confidence and (args.feedback_config or args.paper_fit or
                                   args.radial_restore or args.negative_only or
                                   args.resume_result or args.baseline_result):
        parser.error("Sampled confidence requires a separate fresh author-code run")
    if args.feedback_disabled and not args.feedback_config:
        parser.error("--feedback-disabled requires --feedback-config")
    if args.radial_restore and (args.feedback_config or args.paper_fit or
                               args.resume_result or args.negative_only or
                               args.baseline_result):
        parser.error("Radial geometry must be tested as a separate intervention")
    if args.negative_only and (args.feedback_config or args.paper_fit or args.resume_result):
        parser.error("Negative-only ablation requires a fresh author-code run")
    if args.feedback_config and (args.paper_fit or args.resume_result or args.baseline_result):
        parser.error("Feedback prototype requires a fresh author-code run")
    if args.group_timeout_seconds < 0 or args.resume_elapsed_seconds < 0:
        parser.error("Timeout and previous elapsed seconds must be nonnegative; zero timeout disables it")
    if args.resume_result and (args.baseline_result or args.diagnostic_group or args.profile_steps):
        parser.error("Resume is only for an unprofiled paired evaluation")
    if args.profile_steps < 0 or args.profile_start_step < 0:
        parser.error("Profiling step counts must be nonnegative")
    if args.diagnostic_group and args.baseline_result:
        parser.error("Diagnostic runs cannot reuse formal baselines")
    if args.profile_only and (not args.diagnostic_group or not args.profile_steps):
        parser.error("--profile-only requires --diagnostic-group and positive --profile-steps")
    return args


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
    args.layer, calibration_source_layer = resolve_calibration_layer(
        fitted if fitted is not None else paper, args.model, args.layer
    )
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
    prompts = [build_prompt(tokenizer, row["problem"]) for row in examples]
    max_prompt_tokens = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    if max_prompt_tokens + args.max_tokens > args.max_model_len:
        raise ValueError("Context limit would shorten at least one requested generation")
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
    if args.negative_only:
        dynamic_params["negative_only"] = True
    if args.sampled_confidence:
        dynamic_params["sampled_confidence"] = True
    algorithm = "rebalance"
    if args.radial_restore:
        algorithm = ("rebalance_radial" if args.radial_restore == "on" else
                     "rebalance_radial_disabled")
    feedback = None
    if args.feedback_config:
        import numpy as np
        from vllm.steer_vectors.payloads import FeedbackDirection
        feedback = json.loads(args.feedback_config.read_text(encoding="utf-8"))
        readout_path = args.feedback_config.resolve().parent / feedback["readout_file"]
        if hashlib.sha256(readout_path.read_bytes()).hexdigest() != feedback["readout_sha256"]:
            raise ValueError("Feedback readout hash mismatch")
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != feedback["vector_sha256"]:
            raise ValueError("Feedback requires its fixed original direction")
        if feedback["decoder_output_layer"] != args.layer:
            raise ValueError("Feedback readout layer mismatch")
        payload = FeedbackDirection(
            payload.layers[args.layer], np.load(readout_path, allow_pickle=False),
            feedback["negative_centroid_score"], layer=args.layer,
            enabled=not args.feedback_disabled,
        )
        algorithm = "rebalance_feedback"
    if paper is not None:
        dynamic_params["paper_parameters"] = paper["parameters"]["paper_parameters"]
    steering = SteeringSpec(
        vectors=[
            VectorSpec(
                name="rebalance_dynamic",
                data=payload,
                algorithm=algorithm,
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
            "Arithmetic mean of raw probabilities of the actually sampled "
            "tokens, before temperature, top-p, penalties or grammar; "
            "two-step variance uses these means. Experimental sampled-confidence "
            "variant of the self-calibrated public-code adaptation."
            if args.sampled_confidence else
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
            "rebalance_source_layer": calibration_source_layer,
            "easysteer_output_layer": args.layer,
            "effective_hidden_state_index": args.layer + 1,
            "calibration_layer_verified": calibration_source_layer is not None,
            "negative_only": args.negative_only,
            "sampled_confidence": args.sampled_confidence,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "chunked_prefill": args.chunked_prefill,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "execution_mode": "in_graph",
            "steering_algorithm": algorithm,
            "radial_restore": args.radial_restore,
            "async_scheduling": args.async_scheduling,
            "profiling_enabled": args.profile_steps > 0,
            "profile_steps": args.profile_steps,
            "profile_start_step": args.profile_start_step,
            "profile_only": args.profile_only,
            "length_policy": "all generated tokens, including capped/incorrect answers",
            "dynamic_params": dynamic_params,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "triton": package_version("triton"),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            "numeric_environment": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "compile_environment": {
                    key: os.environ.get(key) for key in (
                        "TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "VLLM_USE_AOT_COMPILE",
                        "VLLM_USE_STANDALONE_COMPILE", "CUDA_VISIBLE_DEVICES",
                    )
                },
            },
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
    if feedback is not None:
        result["feedback"] = feedback
        result["protocol"]["steering_algorithm"] = algorithm
        result["protocol"]["feedback_enabled"] = not args.feedback_disabled
        result["provenance"]["feedback_config_sha256"] = hashlib.sha256(
            args.feedback_config.read_bytes()).hexdigest()
    result["protocol"]["preemption_policy"] = "restore controller and replay historical input scales"
    result["protocol"]["run_order"] = (
        ["rebalance_dynamic", "baseline"] if args.dynamic_first
        else ["baseline", "rebalance_dynamic"])
    result["protocol"]["diagnostic_only"] = args.diagnostic_group is not None
    if args.diagnostic_group:
        result["protocol"]["run_order"] = [args.diagnostic_group]
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

    identity_started = time.perf_counter()
    if Path(vllm.__file__).resolve() != (
        root / "sources/EasySteer/vllm-steer/vllm/__init__.py"
    ).resolve() or Path(inspect.getfile(from_pt_direction)).resolve() != (
        root / "sources/EasySteer/easysteer/vectors.py"
    ).resolve():
        raise ValueError("Imported runtime differs from the source being fingerprinted")
    result["provenance"]["execution_identity"] = execution_identity(root, model_path)
    result["execution_identity_seconds"] = time.perf_counter() - identity_started

    reused_baseline = None
    if args.baseline_result:
        saved_bytes = args.baseline_result.read_bytes()
        saved = json.loads(saved_bytes)
        reused_baseline = validate_saved_baseline(saved, result, examples, think_end_id)
        result["provenance"]["reused_baseline"] = {
            "path": str(args.baseline_result),
            "sha256": hashlib.sha256(saved_bytes).hexdigest(),
            "provenance": saved["provenance"],
            "timing_is_historical": True,
        }

    resumed = None
    if args.resume_result:
        from resume_validation import validate_resume
        resumed, receipt = validate_resume(args.resume_result, output_path, result, args.resume_elapsed_seconds)
        result["provenance"]["resumed_evaluation"] = receipt

    def run_group(prompts, sampling, boundary_set, steering, resume_path=None):
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation group exceeded its time budget")
        previous = signal.signal(signal.SIGALRM, timeout_handler)
        started = time.perf_counter()
        previous_preemptions = preemptions["events"]
        previous_replays = dict(replay_state.replay_counts)
        feedback_tables = []
        if feedback is not None:
            for module in runner.steer_vector_manager.graph_state.controllers:
                tables = getattr(module, "graph_tables", None) or {}
                if "feedback" in tables:
                    feedback_tables.append(tables["feedback"]["stats"])
            if not feedback_tables:
                raise RuntimeError("Feedback graph family was not installed")
            for table in feedback_tables:
                table.zero_()
        replay_state.positive_suppression_counts.zero_()
        # Graders may use SIGALRM internally; keep an independent wall-time cap.
        def enforce_deadline():
            print("Evaluation group exceeded its wall-time budget", file=sys.stderr, flush=True)
            os._exit(124)
        watchdog = threading.Timer(args.group_timeout_seconds, enforce_deadline)
        watchdog.daemon = True
        if args.group_timeout_seconds:
            watchdog.start()
            signal.alarm(args.group_timeout_seconds)
        try:
            step_profiler = None
            if args.profile_steps:
                from decode_profiler import EngineStepProfiler
                step_profiler = EngineStepProfiler(
                    output_path.with_suffix(".baseline.trace.json" if steering is None
                                            else ".dynamic.trace.json"),
                    args.profile_start_step, args.profile_steps,
                    stop_after_window=args.profile_only)
            records, seconds = generate_records(
                llm, prompts, examples, sampling, boundary_set, steering=steering,
                checkpoint_path=output_path.with_suffix(
                    ".baseline.partial.jsonl" if steering is None
                    else ".dynamic.partial.jsonl"),
                step_profiler=step_profiler,
                resume_path=resume_path,
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
            summary["timing_includes_profiler_overhead"] = args.profile_steps > 0
            summary["preemptions"] = preemptions["events"] - previous_preemptions
            summary["dynamic_kv_replay"] = {
                key: value - previous_replays[key]
                for key, value in replay_state.replay_counts.items()
            }
            if args.negative_only:
                counts = replay_state.positive_suppression_counts.cpu().tolist()
                summary["positive_suppression"] = dict(
                    boundary_updates_cancelled=int(counts[0]),
                    positive_coefficient_sum=counts[1],
                    scope="Completed in-think boundary updates; a final capped token may not receive a subsequent forward",
                )
            if feedback_tables:
                totals = torch.stack([t.sum(0) for t in feedback_tables]).sum(0).cpu().tolist()
                summary["feedback_applications"] = dict(zip(
                    ["negative_executions", "coefficient_changed", "negative_cancelled",
                     "requested_absolute_coefficient_sum", "applied_absolute_coefficient_sum"],
                    totals,
                ))
                summary["feedback_counter_scope"] = "All token executions, including any KV replay; coefficient rounding included; amplitude sums are FP32 diagnostics"
            if resume_path is not None:
                summary["resumed_completed_answers"] = sum(1 for _ in resume_path.open())
                summary["prior_interrupted_seconds"] = args.resume_elapsed_seconds
                summary["generation_seconds"] += args.resume_elapsed_seconds
                summary["tokens_per_second"] = summary["total_tokens"] / summary["generation_seconds"]
                summary["group_seconds_including_grading"] += args.resume_elapsed_seconds
                summary["timing_note"] = "Includes prior interrupted arm wall time and discarded unfinished generation; not a single uninterrupted speed measurement"
                summary["resume_session_preemptions"] = summary["preemptions"]
                summary["preemptions"] = None  # Prior interrupted process did not save its counter.
            if steering is not None:
                assert not replay_state._suspended
                assert (summary["dynamic_kv_replay"]["suspended"]
                        == summary["dynamic_kv_replay"]["restored"])
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
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_steer_vector=True,
            steer_algorithms=(["rebalance_radial", "rebalance_radial_disabled"]
                              if args.radial_restore else [algorithm]),
            enforce_eager=False,
            steer_graph_mode="in_graph",
            enable_chunked_prefill=args.chunked_prefill,
            enable_prefix_caching=False,
            async_scheduling=args.async_scheduling,
            seed=args.seed,
        )
        result["startup_seconds"] = time.perf_counter() - started
        core = llm.llm_engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        replay_state = runner.steer_vector_state
        if args.profile_steps:
            from decode_profiler import mark_runtime_ranges
            mark_runtime_ranges(runner)
        if not replay_state.supports_kv_replay:
            raise RuntimeError("This evaluator requires the V2 steering replay runner")
        preemptions = guard_dynamic_preemption(core.scheduler, replay_state)
        result["environment"]["gpu"] = torch.cuda.get_device_name(0)
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            skip_special_tokens=True,
        )
        boundary_set = set(boundary_ids)
        result["status"] = "incomplete"
        write_result(output_path, result)

        for mode in result["protocol"]["run_order"]:
            if resumed is not None and mode in resumed:
                print(f"Retaining completed {mode}; no generation repeat")
                result[mode] = resumed[mode]
            elif mode == "baseline" and reused_baseline is not None:
                print("Reusing verified saved baseline; no baseline generation")
                result[mode] = reused_baseline
            else:
                print(f"Running {mode}...")
                try:
                    resume_path = None
                    if resumed is not None:
                        candidate = args.resume_result.with_suffix(
                            ".baseline.partial.jsonl" if mode == "baseline" else ".dynamic.partial.jsonl")
                        if candidate.exists():
                            resume_path = candidate
                    records, summary = run_group(
                        prompts, sampling, boundary_set,
                        steering=None if mode == "baseline" else steering,
                        resume_path=resume_path,
                    )
                except ProfileWindowComplete as completed:
                    result.update(status="profile_completed", profile_trace=str(completed),
                                  profile_scope="bounded decode window; no full generation or accuracy metrics")
                    write_result(output_path, result)
                    print(f"Profile window saved: {completed}; remaining generation stopped")
                    return
                result[mode] = {"summary": summary, "records": records}
            write_result(output_path, result)
        if args.diagnostic_group:
            result["status"] = "diagnostic_completed"
            write_result(output_path, result)
            print(f"Engineering diagnostic completed: {args.diagnostic_group}; {output_path}")
            return
        baseline, dynamic = result["baseline"]["records"], result["rebalance_dynamic"]["records"]
        baseline_summary = result["baseline"]["summary"]
        dynamic_summary = result["rebalance_dynamic"]["summary"]
        result["comparison"] = comparison(
            baseline,
            dynamic,
            baseline_summary,
            dynamic_summary,
            args.offset,
        )
        result["status"] = "completed"
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
