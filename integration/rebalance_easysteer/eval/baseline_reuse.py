"""Verify reusable controls before loading the model; no torch or CUDA imports."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess

from token_metrics import length_metrics


SOURCE_ROOTS = (
    "integration/rebalance_easysteer/eval",
    "sources/EasySteer/easysteer",
    "sources/EasySteer/vllm-steer/vllm",
    "sources/EasySteer/vllm-steer/csrc",
    "sources/ReBalance",
)
SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
MODEL_EXTENSIONS = {
    ".json", ".safetensors", ".bin", ".pt", ".model", ".txt", ".tiktoken",
    ".jinja", ".py",
}
REQUIRED_SOURCES = (
    "integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py",
    "integration/rebalance_easysteer/eval/rebalance_static_eval.py",
    "integration/rebalance_easysteer/eval/runtime_guards.py",
    "sources/EasySteer/easysteer/vectors.py",
    "sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/model_runner.py",
    "sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/steer_vector_utils.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_identity(project, model):
    """Hash tracked inference code and every recognized local model asset.

    Read actual bytes each run, with bounded memory; never trust size/mtime as
    content identity. Source hashes normalize CRLF to Git's canonical LF. This
    is preparation time, never included in pure generation time. Dependency and
    GPU versions are checked separately in the result's environment metadata.
    """
    project, model = Path(project), Path(model)
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *SOURCE_ROOTS], cwd=project
    )
    names = sorted(name for name in raw.decode("utf-8").split("\0")
                   if name and Path(name).suffix in SOURCE_EXTENSIONS)
    _require(set(REQUIRED_SOURCES).issubset(names), "Inference source inventory missing")
    sources = {
        name: hashlib.sha256((project / name).read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest() for name in names
    }
    files = sorted(path for path in model.rglob("*")
                   if path.is_file() and path.suffix in MODEL_EXTENSIONS
                   and not any(part.startswith(".") for part in path.relative_to(model).parts))
    assets = {path.relative_to(model).as_posix(): file_sha256(path) for path in files}
    _require("config.json" in assets, "Missing model config identity")
    _require(any(Path(name).suffix in {".safetensors", ".bin", ".pt"}
                 for name in assets), "Missing model weight identity")
    _require(any("tokenizer" in name or name.endswith(".model") for name in assets),
             "Missing tokenizer identity")
    return {"version": "inference-content-v1", "source_sha256": sources,
            "model_files_sha256": assets}


def validate_saved_baseline(saved, current, examples, think_end_id):
    """Reject uncertain compatibility. Historical artifacts stay untouched."""
    _require(saved.get("status") == "completed", "Saved pair is not completed")
    old, now = saved["protocol"], current["protocol"]
    _require(not old.get("profiling_enabled") and not old.get("diagnostic_only"),
             "Diagnostic output cannot be reused as a formal control")
    _require(not saved["provenance"].get("git_status", "unknown").strip()
             and not current["provenance"].get("git_status", "unknown").strip(),
             "Baseline reuse requires clean recorded worktrees")
    # A docs-only commit may change; actual inference source bytes must not.
    identity = current["provenance"].get("execution_identity")
    _require(isinstance(identity, dict) and identity.get("version") == "inference-content-v1"
             and identity.get("source_sha256") and identity.get("model_files_sha256"),
             "Current run has no verified content identity")
    _require(saved["provenance"].get("execution_identity") == identity,
             "Saved model/source identity is missing or changed; preserve legacy results")
    for key in ("dataset_sha256", "vector_sha256", "calibration_fit_sha256", "paper_fit_sha256"):
        _require(saved["provenance"].get(key) == current["provenance"].get(key),
                 "Baseline asset changed: " + key)
    # Same inactive steering graph and run order are required as well as sampling.
    for key in ("model", "dataset", "offset", "limit", "max_tokens", "max_model_len",
                "temperature", "top_p", "seed", "execution_mode", "max_num_seqs",
                "gpu_memory_utilization", "async_scheduling", "chunked_prefill",
                "max_num_batched_tokens", "easysteer_output_layer", "steering_algorithm",
                "dynamic_params", "run_order"):
        _require(key in old and key in now and old[key] == now[key],
                 "Baseline protocol changed or missing: " + key)
    for key in ("python", "torch", "cuda", "vllm", "gpu", "transformers", "tokenizers",
                "triton", "cuda_driver", "numeric_environment"):
        _require(key in saved["environment"] and key in current["environment"]
                 and saved["environment"][key] == current["environment"][key],
                 "Baseline environment changed or missing: " + key)
    group = saved.get("baseline", {})
    records = group.get("records", [])
    _require(len(records) == len(examples) > 0, "Missing baseline records")
    cap = now["max_tokens"]
    for index, (row, example) in enumerate(zip(records, examples, strict=True)):
        _require(row.get("dataset_index") == index and row.get("problem") == example["problem"]
                 and row.get("gold") == example["answer"], "Baseline question/gold order mismatch")
        ids = row.get("token_ids")
        _require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids),
                 "Invalid baseline token IDs")
        _require(type(row.get("tokens")) is int and row["tokens"] == len(ids) <= cap,
                 "Baseline token count/cap mismatch")
        _require(row.get("finish_reason") in ("stop", "length"), "Unfinished baseline answer")
        _require(row["finish_reason"] != "length" or len(ids) == cap,
                 "Truncated baseline did not reach the declared cap")
        ended = think_end_id in ids
        thinking = ids.index(think_end_id) if ended else len(ids)
        _require(row.get("thinking_tokens") == thinking and row.get("thinking_ended") is ended
                 and row.get("answer_tokens") == len(ids) - thinking - int(ended),
                 "Baseline thinking/answer token accounting mismatch")
        _require(type(row.get("correct")) is bool, "Invalid baseline correctness value")
    summary = group.get("summary", {})
    expected = dict(length_metrics(records, cap), examples=len(records),
                    correct=sum(row["correct"] for row in records),
                    accuracy=sum(row["correct"] for row in records) / len(records),
                    mean_thinking_tokens=sum(row["thinking_tokens"] for row in records) / len(records),
                    mean_answer_tokens=sum(row["answer_tokens"] for row in records) / len(records),
                    thinking_not_ended=sum(not row["thinking_ended"] for row in records))
    for key, value in expected.items():
        _require(summary.get(key) == value, "Baseline summary/records mismatch: " + key)
    seconds = summary.get("generation_seconds")
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
             "Missing or invalid historical generation time")
    return group
