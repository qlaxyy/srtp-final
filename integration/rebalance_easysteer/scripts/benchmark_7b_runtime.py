"""Two purposeful runtime trials on calibration questions; no test-set tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = "/root/autodl-tmp/venvs/easysteer-vllm026/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare(artifacts, output):
    fit_path = artifacts / "fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("version") != "auto-code-v2" or not fit["model"].endswith("DeepSeek-R1-Distill-Qwen-7B"):
        raise ValueError("Requires the frozen 7B auto-code-v2 calibration")
    vector = artifacts / "auto_vector.pt"
    if sha(vector) != fit["vector_sha256"]:
        raise ValueError("Vector hash mismatch")
    source = artifacts / "calibration/generations.jsonl"
    if sha(source) != fit["calibration_source_sha256"]:
        raise ValueError("Calibration answer hash mismatch")
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    train_path = ROOT / "sources/ReBalance/Data/Math_Train/test.jsonl"
    if sha(train_path) != manifest["train_sha256"]:
        raise ValueError("Calibration training dataset changed")
    train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
    selected = manifest["train_indices"][:64]
    if len(set(selected)) != 64:
        raise ValueError("Need 64 unique calibration questions")
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "calibration64.jsonl"
    dataset.write_text("".join(json.dumps(train[i], ensure_ascii=False) + "\n" for i in selected),
                       encoding="utf-8")
    plan = dict(status="prepared_not_run", purpose="仅比较运行并发，固定模型/向量/控制函数/生成上限",
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                fit_sha256=sha(fit_path), vector_sha256=sha(vector),
                calibration_sha256=sha(source), dataset_sha256=sha(dataset),
                train_indices=selected, accuracy_is_diagnostic_only=True, runs=[])
    for concurrency in (32, 64):
        result = output / f"concurrency{concurrency}.json"
        command = [PYTHON, "-u", str(ROOT / "integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py"),
                   "--model", fit["model"], "--dataset", str(dataset), "--limit", "64",
                   "--max-tokens", "16000", "--max-model-len", "17408",
                   "--temperature", "0", "--top-p", "1", "--seed", "42",
                   "--max-num-seqs", str(concurrency), "--chunked-prefill",
                   "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.92",
                   "--vector", str(vector), "--calibration-fit", str(fit_path),
                   "--diagnostic-group", "rebalance_dynamic",
                   "--group-timeout-seconds", "1500", "--output", str(result)]
        plan["runs"].append(dict(name=f"并发{concurrency}", status="not_started", command=command,
                                 features="64道相同校准题，贪心，上限16000，动态ReBalance，非正式测评",
                                 result=str(result), max_num_seqs=concurrency))
    save(output / "run_ledger.json", plan)
    return plan


def compare_results(first, second):
    if (first["protocol"]["max_num_seqs"], second["protocol"]["max_num_seqs"]) != (32, 64):
        raise ValueError("Expected concurrency32 then concurrency64")
    for result in (first, second):
        if result["status"] != "diagnostic_completed":
            raise ValueError("Incomplete diagnostic")
        if len(result["rebalance_dynamic"]["records"]) != 64:
            raise ValueError("Incomplete calibration subset")
    # Concurrency alone may differ. No profiler overhead in speed measurements.
    for key in ("model", "dataset", "max_tokens", "max_model_len", "temperature", "top_p",
                "seed", "dynamic_params", "easysteer_output_layer", "chunked_prefill",
                "max_num_batched_tokens", "gpu_memory_utilization", "async_scheduling"):
        if first["protocol"][key] != second["protocol"][key]:
            raise ValueError(f"Comparison changed another setting: {key}")
    for result in (first, second):
        if result["protocol"].get("profiling_enabled"):
            raise ValueError("Cannot report profiling runs as speed evidence")
    for key in ("dataset_sha256", "vector_sha256", "calibration_fit_sha256", "commit"):
        if first["provenance"][key] != second["provenance"][key]:
            raise ValueError(f"Different provenance: {key}")
    a, b = first["rebalance_dynamic"], second["rebalance_dynamic"]
    differences = []
    for index, (x, y) in enumerate(zip(a["records"], b["records"], strict=True)):
        if x["problem"] != y["problem"]:
            raise ValueError("Question order changed")
        if x["token_ids"] != y["token_ids"]:
            position = next((i for i, (u, v) in enumerate(zip(x["token_ids"], y["token_ids"]))
                             if u != v), min(len(x["token_ids"]), len(y["token_ids"])))
            differences.append(dict(index=index, first_different_token=position,
                                    tokens32=len(x["token_ids"]), tokens64=len(y["token_ids"])))
    return dict(output_equal=not differences, differences=differences,
                generation_speed_ratio32_over64=a["summary"]["generation_seconds"] / b["summary"]["generation_seconds"],
                summaries={"concurrency32": a["summary"], "concurrency64": b["summary"]},
                decision="待核对吞吐和缓存回收；有输出差异时先定位，不自动更改正式配置",
                formal_accuracy_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Actually run two GPU trials; default prepares only")
    args = parser.parse_args()
    if args.run and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit or resolve workspace changes before timed GPU trials")
    out = args.output.resolve()
    plan = prepare(args.artifacts.resolve(), out)
    print(f"Prepared two calibration diagnostics: {out}", flush=True)
    if not args.run:
        return
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    env["PATH"] = str(Path(PYTHON).parent) + os.pathsep + env.get("PATH", "")
    for job in plan["runs"]:
        job.update(status="running", started=time.time())
        plan["status"] = "running"
        save(out / "run_ledger.json", plan)
        print(job["name"] + ": " + job["features"], flush=True)
        try:
            with Path(job["result"]).with_suffix(".log").open("w", encoding="utf-8") as log:
                run = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=1620)
            if run.returncode:
                raise RuntimeError(f"Evaluation exited {run.returncode}; partial answers retained")
        except Exception as exc:
            job.update(status="failed", error=str(exc), finished=time.time())
            plan["status"] = "failed"
            save(out / "run_ledger.json", plan)
            raise
        job.update(status="completed", finished=time.time())
        save(out / "run_ledger.json", plan)
    results = [json.loads(Path(job["result"]).read_text(encoding="utf-8")) for job in plan["runs"]]
    comparison = compare_results(*results)
    save(out / "comparison.json", comparison)
    plan["status"] = "completed_needs_review"
    save(out / "run_ledger.json", plan)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
