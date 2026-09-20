"""Audit the completed engineering archive locally; no inference or grading."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite audit"
    raw = args.raw
    manifest = read(raw / "artifact_manifest.json")
    for relative, digest in manifest.items():
        path = (raw / relative).resolve()
        assert path.is_relative_to(raw.resolve())
        assert sha(path) == digest, relative
    arms = ["R", "Roff_history", "RC14default", "RC14explicit",
            "RChistory_shadow", "RChistory"]
    data = {arm: read(raw / "results" / arm / "result.json") for arm in arms}
    plan = read(raw / "results/resolved_plan.json")
    for arm, d in data.items():
        assert d["status"] == "complete" and len(d["records"]) == 8
        assert d["replay_counts"] == {"suspended": 1, "restored": 1}
        assert d["forced_preemption"] and d["extra_probe_forwards"] == 0
        for record, expected in zip(d["records"], plan["engineering_rows"]):
            assert record["train_index"] == expected["train_index"]
            assert record["problem_sha256"] == expected["problem_sha256"]
            assert len(record["token_ids"]) == record["tokens"] == 512
    for left, right in [("R", "Roff_history"), ("R", "RChistory_shadow"),
                        ("RC14default", "RC14explicit")]:
        for a, b in zip(data[left]["records"], data[right]["records"]):
            assert a["token_ids"] == b["token_ids"]
            assert a["R_history_sha256"] == b["R_history_sha256"]
    rows = []
    for i, candidate in enumerate(data["RChistory"]["records"]):
        event = data["RChistory"]["events"][candidate["request_id"]]
        if event["changed"]:
            assert 0 <= event["first_reflection"] < event["first_change"] < 512
        row = {key: candidate[key] for key in ["train_index", "problem_sha256"]}
        row["events"] = event
        for other in ["R", "RC14explicit"]:
            reference = data[other]["records"][i]
            row["different_from_" + other] = candidate["token_ids"] != reference["token_ids"]
            row["native_history_equal_" + other] = candidate["R_history_sha256"] == reference["R_history_sha256"]
            row["first_difference_" + other] = next((j for j, (a, b) in enumerate(
                zip(candidate["token_ids"], reference["token_ids"])) if a != b), None)
        rows.append(row)
    assert any(row["different_from_RC14explicit"] for row in rows)
    tests = {}
    for name in ["test_native.py", "test_narrow_native.py", "test_history_native.py"]:
        tests[name] = read(raw / "results" / (name + ".json"))
        assert tests[name]["returncode"] == 0
    gate = read(raw / "results/engineering_gate.json")
    assert gate["passed"]
    assert gate["plan_sha256"] == sha(raw / "results/resolved_plan.json")
    exit_receipt = read(raw / "launch/history_exit.json")
    assert exit_receipt["returncode"] == 0
    summary = {
        "status": "engineering_passed_effect_unverified_formal_not_authorized",
        "raw_root": str(raw.resolve()), "verified_manifest_files": len(manifest),
        "execution_commit": read(raw / "closure.json")["execution_commit"],
        "plan_sha256": gate["plan_sha256"], "questions": 8, "arms": 6,
        "cap": 512, "answers": 48, "total_generated_tokens": 24576,
        "pure_generation_seconds": sum(d["generation_seconds"] for d in data.values()),
        "runner_wall_seconds": gate["wall_seconds"],
        "process_wall_seconds": exit_receipt["wall_seconds"],
        "arm_metrics": {arm: {**{key: d[key] for key in ["generation_seconds",
            "setup_seconds", "checkpoint_io_seconds", "replay_counts",
            "extra_probe_forwards", "control_gpu_seconds"]},
            "logit_mask_applications": sum(e["changed"] for e in d["events"].values())}
            for arm, d in data.items()},
        "different_from_RC14": sum(r["different_from_RC14explicit"] for r in rows),
        "different_from_R": sum(r["different_from_R"] for r in rows),
        "questions_observing_first_marker": sum(r["events"]["first_reflection"] >= 0 for r in rows),
        "per_question": rows,
        "limits": ["Engineering cap512 outputs are not formal accuracy or compression evidence.",
            "Mask applications count altered logits, not altered sampled tokens.",
            "All eight candidate token sequences equal R in this short window.",
            "No independent control-kernel timing was collected; generation includes its overhead.",
            "Forced replay is engineering coverage; these timings do not establish production speed.",
            "No fresh100 generation; reserved confirmation200 untouched."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["per_question", "arm_metrics", "limits"]}))


if __name__ == "__main__":
    main()
