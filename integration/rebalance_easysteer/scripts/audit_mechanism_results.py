"""Independently recount saved screen outputs on CPU, without grading or inference.

This post-run audit does not import the experiment runner or its comparison code.
The downloaded files and the fixed pre-run plan are read-only inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path, source=False):
    data = path.read_bytes()
    if source:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    bundle, results = args.bundle.resolve(), args.results.resolve()
    check(not args.receipt.exists(), "Receipt already exists; inspect it")
    root = Path(__file__).resolve().parents[3]
    plan, ledger = read(bundle / "plan.json"), read(results / "run_ledger.json")
    analysis, manifest = read(results / "analysis.json"), read(results / "export_manifest.json")
    rows = [json.loads(line) for line in (bundle / "screen100.jsonl").read_text(encoding="utf-8").splitlines()]
    check(len(rows) == plan["count"] == 100, "Unexpected screen count")
    check([r["train_index"] for r in rows] == plan["train_indices"], "Dataset order changed")
    check(digest(bundle / "plan.json") == ledger["plan_sha256"], "Plan changed")
    check(digest(bundle / "screen100.jsonl") == plan["dataset_sha256"], "Dataset changed")
    check(analysis["status"] == "completed", "Final analysis incomplete")
    for name, item in manifest["files"].items():
        check(Path(name).name == name, "Unexpected export path")
        check((results / name).stat().st_size == item["bytes"], "Size mismatch: " + name)
        check(digest(results / name) == item["sha256"], "Export hash mismatch: " + name)
    for name, expected in analysis["source_sha256"].items():
        check(digest(results / name) == expected, "Analysis input changed: " + name)
    for name, expected in plan["source_sha256"].items():
        check(digest(root / name, source=True) == expected, "Runtime source changed: " + name)

    raw, grades, partials, groups = {}, {}, {}, {}
    for arm in plan["run_order"]:
        data, grade = read(results / (arm + ".json")), read(results / (arm + ".author.json"))
        check(data["status"] == "diagnostic_completed" and grade["status"] == "completed", "Incomplete arm")
        sha = digest(results / (arm + ".json"))
        check(sha == ledger["arms"][arm]["raw_sha256"] == grade["input_sha256"], "Raw/grade link differs")
        check(data["provenance"]["commit"] == ledger["commit"] and not data["provenance"]["git_status"], "Dirty/different run commit")
        check(data["provenance"]["dataset_sha256"] == grade["dataset_sha256"] == plan["dataset_sha256"], "Different dataset")
        asset = plan["arms"][arm]
        for filename, key in [("auto_vector.pt", "vector_sha256"), ("fit.json", "fit_sha256")]:
            check(digest(bundle / asset["directory"] / filename) == asset[key], "Candidate asset changed")
        check(data["provenance"]["vector_sha256"] == asset["vector_sha256"], "Wrong runtime vector")
        check(data["provenance"]["calibration_fit_sha256"] == asset["fit_sha256"], "Wrong runtime fit")
        for name, expected in grade["grader_sources"].items():
            check(digest(root / "sources/ReBalance/utils" / name, source=True) == expected, "Author grader changed")
        check(read(results / (arm + ".author.partial.json"))["records"] == grade["records"], "Author partial differs")
        parts = [json.loads(line) for line in (results / (arm + ".dynamic.partial.jsonl")).read_text(encoding="utf-8").splitlines()]
        check(len(parts) == 100 and sorted(p["local_index"] for p in parts) == list(range(100)), "Partial ids incomplete/duplicated")
        parts = {p["local_index"]: p for p in parts}
        records = data["rebalance_dynamic"]["records"]
        check(len(records) == len(grade["records"]) == 100, "Missing records")
        protocol = data["protocol"]
        for key, value in plan["runtime"].items():
            check(protocol[key] == value, "Runtime mismatch: " + key)
        check(protocol["max_tokens"] == 16000, "Cap changed")
        end = protocol["dynamic_params"]["think_end_token_id"]
        boundary_ids = set(protocol["dynamic_params"]["boundary_token_ids"])
        for i, (row, record, score) in enumerate(zip(rows, records, grade["records"])):
            check(record["dataset_index"] == score["index"] == i and score["train_index"] == row["train_index"], "Pairing mismatch")
            check(record["problem"] == row["problem"], "Question changed")
            check(type(score["correct"]) is bool, "Nonboolean author score")
            ids, part = record["token_ids"], parts[i]
            check(0 < len(ids) == record["tokens"] <= 16000, "Invalid token count")
            check(ids == part["token_ids"] and record["text"] == part["text"] and record["finish_reason"] == part["finish_reason"], "Partial/full output mismatch")
            ended = end in ids
            thinking = ids.index(end) if ended else len(ids)
            check(record["thinking_ended"] == ended and record["thinking_tokens"] == thinking, "Thinking count mismatch")
            check(record["answer_tokens"] == (len(ids) - thinking - 1 if ended else 0), "Answer count mismatch")
            check(record["boundary_tokens"] == sum(t in boundary_ids for t in ids), "Boundary count mismatch")
        totals = [len(r["token_ids"]) for r in records]
        stats = dict(count=100, correct=sum(g["correct"] for g in grade["records"]),
                     mean_thinking_tokens=sum(r["thinking_tokens"] for r in records) / 100,
                     mean_total_tokens=sum(totals) / 100, total_tokens=sum(totals),
                     capped=sum(r["finish_reason"] == "length" or len(r["token_ids"]) == 16000 for r in records),
                     thinking_not_ended=sum(not r["thinking_ended"] for r in records),
                     math_verify_correct=sum(r["correct"] for r in records),
                     math_verify_author_disagreement_indices=[i for i, (r, g) in enumerate(zip(records, grade["records"])) if r["correct"] != g["correct"]],
                     generation_seconds=data["rebalance_dynamic"]["summary"]["generation_seconds"],
                     startup_seconds=data["startup_seconds"], process_seconds=ledger["arms"][arm]["process_seconds"],
                     author_grading_seconds=grade["seconds"])
        check(stats["correct"] == grade["correct"], "Author total mismatch")
        summary = data["rebalance_dynamic"]["summary"]
        for key in ["mean_thinking_tokens", "total_tokens", "capped", "thinking_not_ended"]:
            check(stats[key] == summary[key], "Raw summary mismatch: " + key)
        check(stats["mean_total_tokens"] == summary["mean_tokens"] and stats["math_verify_correct"] == summary["correct"], "Math-verify summary mismatch")
        raw[arm], grades[arm], partials[arm], groups[arm] = data, grade, parts, stats

    reference = plan["run_order"][0]
    differences = {}
    for arm in plan["run_order"][1:]:
        a, b = raw[reference], raw[arm]
        check({k: v for k, v in a["protocol"].items() if k != "vector"} == {k: v for k, v in b["protocol"].items() if k != "vector"}, "Protocol differs beyond vector")
        check(a["environment"] == b["environment"] and a["calibration"]["parameters"] == b["calibration"]["parameters"], "Runtime/calibration differs")
        check(all(partials[reference][i]["prompt_token_ids"] == partials[arm][i]["prompt_token_ids"] for i in range(100)), "Prompt token ids differ")
        paired = []
        for i, (x, y, gx, gy) in enumerate(zip(a["rebalance_dynamic"]["records"], b["rebalance_dynamic"]["records"], grades[reference]["records"], grades[arm]["records"])):
            paired.append(dict(index=i, train_index=rows[i]["train_index"], original_correct=gx["correct"], candidate_correct=gy["correct"],
                               thinking_token_delta=y["thinking_tokens"] - x["thinking_tokens"], total_token_delta=y["tokens"] - x["tokens"],
                               original_capped=x["finish_reason"] == "length" or x["tokens"] == 16000,
                               candidate_capped=y["finish_reason"] == "length" or y["tokens"] == 16000))
        stored = analysis["comparisons"][arm]
        check(paired == stored["per_question"], "Per-question analysis differs")
        gs, rs = groups[arm], groups[reference]
        changes = dict(thinking_token_change_percent=100 * (gs["mean_thinking_tokens"] / rs["mean_thinking_tokens"] - 1),
                       total_token_change_percent=100 * (gs["mean_total_tokens"] / rs["mean_total_tokens"] - 1),
                       accuracy_change_percentage_points=gs["correct"] - rs["correct"])
        for key, value in changes.items():
            check(abs(value - stored[key]) < 1e-10, "Change differs: " + key)
        passes = changes["thinking_token_change_percent"] <= -5 and changes["total_token_change_percent"] <= -5 and gs["correct"] >= rs["correct"] and gs["capped"] <= rs["capped"]
        check(passes == stored["passes_fixed_screen"], "Fixed gate differs")
        differences[arm] = dict(**changes, passes_fixed_screen=passes,
            total_shorter=sum(p["total_token_delta"] < 0 for p in paired), total_longer=sum(p["total_token_delta"] > 0 for p in paired), total_equal=sum(p["total_token_delta"] == 0 for p in paired),
            identical_token_sequences=sum(x["token_ids"] == y["token_ids"] for x, y in zip(a["rebalance_dynamic"]["records"], b["rebalance_dynamic"]["records"])),
            newly_capped_indices=[p["index"] for p in paired if not p["original_capped"] and p["candidate_capped"]],
            no_longer_capped_indices=[p["index"] for p in paired if p["original_capped"] and not p["candidate_capped"]],
            improved_indices=[p["index"] for p in paired if not p["original_correct"] and p["candidate_correct"]],
            degraded_indices=[p["index"] for p in paired if p["original_correct"] and not p["candidate_correct"]])
        for key in ["improved_indices", "degraded_indices"]:
            check(differences[arm][key] == stored[key], "Correctness flips differ")
    check(not any(d["passes_fixed_screen"] for d in differences.values()) and analysis["candidate_for_separate_confirmation"] is None, "Unexpected advancement")
    receipt = dict(status="all_local_recounts_and_input_links_match", scope="Post-run independent audit, no new inference or grading", run_commit=ledger["commit"],
                   script_sha256=digest(Path(__file__), source=True), plan_sha256=digest(bundle / "plan.json"),
                   export_manifest_sha256=digest(results / "export_manifest.json"), analysis_sha256=digest(results / "analysis.json"),
                   files_verified=len(manifest["files"]), runtime_sources_verified=len(plan["source_sha256"]),
                   groups=groups, comparisons=differences, batch_wall_seconds=ledger["completed_unix"] - ledger["started_unix"],
                   pure_generation_seconds=sum(g["generation_seconds"] for g in groups.values()),
                   new_generations=0, author_regrades=0, raw_files_modified=False, confirmation_run=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
