"""CPU-only checks for retaining completed answers from one interrupted pair."""
import hashlib
import json
from pathlib import Path
import subprocess


def validate_resume(source, destination, current, elapsed):
    if source.resolve() == destination.resolve():
        raise ValueError("Resume must preserve the original result")
    old = json.loads(source.read_text(encoding="utf-8"))
    if old.get("status") != "incomplete":
        raise ValueError("Only interrupted pairs need resuming")
    strip = lambda p: {k: v for k, v in p.items() if k != "group_timeout_seconds"}
    if strip(old["protocol"]) != strip(current["protocol"]):
        raise ValueError("Resume protocol changed (only wall-time budget may differ)")
    for key in ("dataset_sha256", "vector_sha256", "calibration_fit_sha256"):
        if old["provenance"].get(key) != current["provenance"].get(key):
            raise ValueError(f"Resume provenance mismatch: {key}")
    for key in ("scope", "confidence_definition", "calibration"):
        if old.get(key) != current.get(key):
            raise ValueError(f"Resume method mismatch: {key}")
    for key in ("python", "torch", "cuda", "vllm"):
        if old["environment"].get(key) != current["environment"].get(key):
            raise ValueError(f"Resume environment mismatch: {key}")
    root = Path(__file__).resolve().parents[3]
    for name in ("steer_vectors/rebalance.py", "v1/worker/gpu/model_runner.py",
                 "v1/worker/gpu/steer_vector_utils.py"):
        path = "sources/EasySteer/vllm-steer/vllm/" + name
        previous = subprocess.check_output(["git", "show", old["provenance"]["commit"] + ":" + path], cwd=root)
        # Git stores LF; a Windows working tree may contain CRLF.
        if previous != (root / path).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Resume inference implementation changed: {path}")
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   original_provenance=old["provenance"], prior_interrupted_seconds=elapsed,
                   checkpoints={}, scheduling_note="Only unfinished questions re-enqueued; batching changes at restart, completed answers are never regenerated or selected by correctness")
    partial_groups = 0
    for group, suffix in (("baseline", ".baseline.partial.jsonl"),
                          ("rebalance_dynamic", ".dynamic.partial.jsonl")):
        if group in old:
            if len(old[group]["records"]) != current["protocol"]["limit"]:
                raise ValueError("Saved complete arm has missing records")
            continue
        checkpoint = source.with_suffix(suffix)
        if checkpoint.exists():
            raw = checkpoint.read_bytes()
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            ids = [r["local_index"] for r in rows]
            if len(set(ids)) != len(ids) or any(not 0 <= i < current["protocol"]["limit"] for i in ids):
                raise ValueError("Checkpoint has invalid question indices")
            partial_groups += 1
            receipt["checkpoints"][group] = dict(path=str(checkpoint), sha256=hashlib.sha256(raw).hexdigest(), retained=len(rows))
    if partial_groups != 1 or elapsed <= 0:
        raise ValueError("Exactly one interrupted checkpoint and its elapsed cost are required")
    return old, receipt
