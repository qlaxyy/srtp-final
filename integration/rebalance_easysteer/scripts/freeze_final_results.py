"""Validate four complete paired benchmarks and freeze their metrics/provenance.

CPU only. Does not generate, regrade, select results, or overwrite a prior lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("baseline", "rebalance_dynamic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_sha(path):
    relative = Path(path).relative_to(ROOT).as_posix()
    blob = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def paired_metrics(directory, dataset, model_size):
    source = directory / f"{dataset}_eval.json"
    grading = directory / f"{dataset}_author_grading.json"
    run, grade = read(source), read(grading)
    protocol, provenance = run["protocol"], run["provenance"]
    count = 500 if dataset == "math500" else 1319
    data = ROOT / "sources/ReBalance/Data" / (
        "Math_Math500" if dataset == "math500" else "Math_GSM8K") / "test.jsonl"
    questions = [json.loads(line)["problem"] for line in data.read_text(encoding="utf-8").splitlines() if line]
    assert len(questions) == count
    assert run.get("status", "completed") == "completed", "Incomplete evaluation"
    assert not protocol.get("diagnostic_only") and not protocol.get("profiling_enabled")
    assert protocol["model"].endswith(f"DeepSeek-R1-Distill-Qwen-{model_size}")
    for key, expected in dict(offset=0, limit=count, max_tokens=16000,
                              temperature=0.7, top_p=0.95, seed=42).items():
        assert protocol[key] == expected, f"Unexpected {key}"
    config = ROOT / "integration/rebalance_easysteer/configs" / (
        "auto_code_v2_1p5b_20260908.json" if model_size == "1.5B"
        else "auto_code_v2_qwen7b_20260908.json")
    fit = read(config)
    assert fit["version"] == "auto-code-v2"
    assert run["calibration"] == fit, "Calibration differs from frozen model parameters"
    assert all(protocol["dynamic_params"][key] == value for key, value in fit["parameters"].items())
    assert provenance["vector_sha256"] == fit["vector_sha256"]
    assert protocol["rebalance_source_layer"] == fit["hidden_state_index"]
    assert protocol["easysteer_output_layer"] == fit["decoder_output_layer"]
    assert grade["input_sha256"] == sha(source), "Grading belongs to a different output"
    assert grade["dataset_sha256"] == provenance["dataset_sha256"] == sha(data)
    for filename in ("grader.py", "parser.py"):
        assert grade[filename + "_sha256"] == sha(ROOT / "sources/ReBalance/utils" / filename)
    groups = {}
    for name in GROUPS:
        records, summary = run[name]["records"], run[name]["summary"]
        scores = grade["groups"][name]
        assert len(records) == len(scores["records"]) == count, "Missing answers"
        assert [r["problem"] for r in records] == questions, "Question order differs"
        assert [r["index"] for r in scores["records"]] == list(range(count))
        total = sum(r["tokens"] for r in records)
        assert all(r["tokens"] == len(r["token_ids"]) and 0 <= r["tokens"] <= 16000 for r in records)
        assert summary["total_tokens"] == total
        assert abs(summary["mean_tokens"] - total / count) < 1e-8
        thinking = sum(r["thinking_tokens"] for r in records) / count
        assert abs(summary["mean_thinking_tokens"] - thinking) < 1e-8
        correct = sum(r["author_correct"] for r in scores["records"])
        assert correct == scores["author_correct"]
        budget = protocol.get("group_timeout_seconds", 1500)
        groups[name] = dict(correct=correct, accuracy_percent=100 * correct / count,
            mean_tokens=total / count, mean_thinking_tokens=thinking,
            capped=sum(r["finish_reason"] == "length" for r in records),
            generation_seconds=summary["generation_seconds"],
            generation_and_grading_seconds=scores["total_group_seconds"],
            resumed_completed_answers=summary.get("resumed_completed_answers", 0),
            prior_interrupted_seconds=summary.get("prior_interrupted_seconds", 0),
            timing_note=summary.get("timing_note"),
            within_original_time_budget=(scores["total_group_seconds"] <= budget) if budget else None,
            preemptions=summary.get("preemptions"))
    base, dynamic = (groups[name] for name in GROUPS)
    return dict(model=model_size, dataset=dataset, count=count, groups=groups,
        delta=dict(accuracy_pp=dynamic["accuracy_percent"] - base["accuracy_percent"],
            total_tokens_percent=100 * (dynamic["mean_tokens"] / base["mean_tokens"] - 1),
            thinking_tokens_percent=100 * (dynamic["mean_thinking_tokens"] / base["mean_thinking_tokens"] - 1)),
        protocol=protocol, provenance=provenance, environment=run["environment"],
        artifacts=dict(evaluation=source.name, evaluation_sha256=sha(source),
            grading=grading.name, grading_sha256=sha(grading), frozen_config=str(config.relative_to(ROOT)),
            frozen_config_sha256=source_sha(config)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onefive", required=True, type=Path)
    parser.add_argument("--seven-gsm", required=True, type=Path)
    parser.add_argument("--seven-math", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit validated scripts before creating the result lock")
    pairs = [(args.onefive, "math500", "1.5B"), (args.onefive, "gsm8k", "1.5B"),
             (args.seven_gsm, "gsm8k", "7B"), (args.seven_math, "math500", "7B")]
    benchmarks = [paired_metrics(*pair) for pair in pairs]
    asset_manifest = args.seven_math / "asset_manifest.json"
    files = sorted((ROOT / "integration/rebalance_easysteer").rglob("*.py"))
    files += sorted((ROOT / "integration/rebalance_easysteer/scripts").glob("*.sh"))
    files += [ROOT / "sources/EasySteer/vllm-steer/vllm" / f for f in (
        "steer_vectors/rebalance.py", "v1/worker/gpu/model_runner.py",
        "v1/worker/gpu/steer_vector_utils.py")]
    lock = dict(status="frozen_complete", name="EasySteer + ReBalance auto-code-v2",
        time_policy="User removed the MATH-500 wall-time limit to complete evaluation; generation cap remains16000",
        scope="Self-calibrated adaptation of released code; not an exact paper reproduction",
        freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        benchmarks=benchmarks,
        assets=read(asset_manifest), asset_manifest_sha256=sha(asset_manifest),
        limitations=["Independent per-model calibration and layer selection; shared algorithm, distinct fitted values",
            "Historical 1.5B uses older runtime and lacks preemption counters; do not assume no preemption",
            "1.5B context32768; 7B context17408; 7B GSM concurrency32 and MATH concurrency64",
            "Do not pool model accuracies or attribute cross-model latency differences to engineering gains",
            "All capped and incorrect generations remain included; single seed"],
        source_hash_format="SHA256 of canonical Git blobs; independent of Windows CRLF checkout",
        source_sha256={p.relative_to(ROOT).as_posix(): source_sha(p) for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, ensure_ascii=False, indent=2)
    for b in benchmarks:
        print(b["model"], b["dataset"], b["delta"])


if __name__ == "__main__":
    main()
