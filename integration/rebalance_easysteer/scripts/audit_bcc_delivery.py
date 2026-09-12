"""Read-only BCC delivery checks; no model imports, generation, or regrading.

Semantic review decisions are supplied by a reviewer, not inferred by this tool.
The tool verifies their exact quotes and keeps core validity separate from errors.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, delivery = args.root.resolve(), args.delivery.resolve()
    require(not args.output.exists(), "Output exists; preserve the previous audit")
    manifest = read(delivery / "recomputed_v2/bcc_candidate_manifest.json")
    require(manifest == read(delivery / "evidence/verified/bcc_candidate_manifest.json"),
            "Candidate manifest differs from supplied one")
    integrity = read(delivery / "recomputed_v2/archive_integrity.json")
    require(integrity == read(delivery / "evidence/verified/archive_integrity.json"),
            "Snapshot integrity receipt differs")
    supplied = read(delivery / "evidence/verified/latest_cpu_audit.json")
    recomputed = read(delivery / "recomputed_v2/latest_cpu_audit.json")
    path_differences = []
    for batch, arms in recomputed["batches"].items():
        for arm, summary in arms.items():
            if not isinstance(summary, dict):
                continue
            for field in ("raw_file", "label_file"):
                if field in summary:
                    value = summary[field]
                    if value != supplied["batches"][batch][arm][field]:
                        path_differences.append([batch, arm, field])
                    summary[field] = value.replace("\\", "/")
                    require(sha(root / summary[field]) == summary[field.replace("file", "sha256")],
                            "Live output/label differs from the audited snapshot")
    require(recomputed == supplied, "Numeric/provenance discrepancy in CPU replication")
    base = root / (".codex_work/overnight_research_20260912/"
                   "prepared_gpu_execution_20260912/unpacked/local_prepared_batch_20260912")
    pairs, review_inputs, review_keys = [], [], []
    for batch, group in sorted(manifest["groups"].items()):
        records = {}
        for arm in ("original_dynamic", batch):
            obj = read(base / batch / "screen" / (arm + ".json"))
            labels = read(base / batch / "screen" / (arm + ".author.json"))
            records[arm] = {label["train_index"]: row for row, label in
                            zip(obj["rebalance_dynamic"]["records"], labels["records"])}
        for p in group["details"]:
            for arm, positions in p["positions"].items():
                row = records[arm][p["train_index"]]
                a, b = positions
                require(p["lcp_tokens"] <= a < b < row["thinking_tokens"], "Invalid capture indices")
                pairs.append(dict(batch=batch, train_index=p["train_index"], arm=arm,
                                  short=arm == p["short_arm"], positions=positions,
                                  distance_from_fork=[i - p["lcp_tokens"] for i in positions],
                                  fraction_of_thinking=[i / row["thinking_tokens"] for i in positions]))
        ordered = sorted(group["details"], key=lambda p: hashlib.sha256(
            f"BCC-v1|20260912|{batch}|{p['train_index']}".encode()).hexdigest())
        for p in ordered:
            if not p["blind_review"]:
                continue
            arms = sorted(records, key=lambda arm: hashlib.sha256(
                f"BCC-review-order|{p['train_index']}|{arm}".encode()).hexdigest())
            i = len(review_inputs) + 1
            review_inputs.append(dict(review_id=i, problem=records[arms[0]][p["train_index"]]["problem"],
                                      traces={label: records[arm][p["train_index"]]["text"]
                                              for label, arm in zip(("A", "B"), arms)}))
            review_keys.append(dict(review_id=i, batch=batch, train_index=p["train_index"],
                                    arms=dict(zip(("A", "B"), arms)), short_arm=p["short_arm"]))
    require(len(pairs) == 152 and len(review_inputs) == 12, "Wrong review/prefix count")
    require(review_inputs == read(delivery / "review_blinded.json"), "Review display not reproducible")
    review = read(args.review)
    require([r["review_id"] for r in review["records"]] == list(range(1, 13)), "Missing semantic reviews")
    citations = []
    for decision, shown, key in zip(review["records"], review_inputs, review_keys):
        for quote in decision.get("evidence", []):
            text = shown["traces"][quote["trace"]]
            require(quote["quote"] in text, "Quoted evidence absent from full trace")
            start = text.index(quote["quote"])
            citations.append(dict(review_id=decision["review_id"], trace=quote["trace"],
                                  character_start=start, character_end=start + len(quote["quote"]),
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  is_short=key["arms"][quote["trace"]] == key["short_arm"]))
    # Exact independent checks of the three material review issues.
    require(Fraction(150, 37) > 4, "Fraction check failed")
    a, b, c = Fraction(4), Fraction(-2), Fraction(-8)
    require([a*b+4*b, b*c+4*c, c*a+4*a] == [-16]*3 and a*b*c == 64 and a != b,
            "Counterexample to all-equal claim failed")
    # Rescale the linear factor by 2 and the quadratic by 1/2.
    coefficients = (Fraction(18), Fraction(4), Fraction(81, 2), Fraction(-9), Fraction(2))
    a, b, c, d, e = coefficients
    require([a*c, a*d+b*c, a*e+b*d, b*e] == [729, 0, 0, 8] and sum(coefficients) != 78,
            "Nonuniqueness witness failed")
    plan_path = root / "integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json"
    plan = read(plan_path)
    exclusions = dict(plan["exclusions"])
    for candidate, stages in plan["splits"].items():
        for stage, indices in stages.items():
            exclusions[candidate + "/" + stage] = indices
    forbidden = set().union(*(set(v) for v in exclusions.values()))
    train_path = root / "sources/ReBalance/Data/Math_Train/test.jsonl"
    train = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normal = lambda s: re.sub(r"\s+", "", s)
    seen = {normal(train[i]["problem"]) for i in forbidden}
    for name, digest in plan["test_prompt_source_sha256"].items():
        data = (root / name).read_bytes().replace(b"\r\n", b"\n")
        require(hashlib.sha256(data).hexdigest() == digest, "Test source changed")
        seen.update(normal(json.loads(line)["problem"]) for line in data.decode().splitlines() if line.strip())
    eligible = []
    for i, row in enumerate(train):
        problem = normal(row["problem"])
        if i not in forbidden and problem not in seen:
            eligible.append(i)
            seen.add(problem)
    require({p["train_index"] for p in pairs} <= forbidden, "Pair missing from exclusions")
    result = dict(status="cpu_audit_complete_content_gate_not_cleared", new_generations=0, GPU_calls=0,
                  replay="not_run", fitting="not_run", screening="not_run", confirmation="not_run",
                  integrity=integrity, windows_path_fields_normalized=path_differences,
                  latest_400_numeric_replication=True, live_raw_and_labels_match=True,
                  pair_manifest_exact_match=True, capture_positions=pairs,
                  capture_positions_at_least_90_percent=sum(v >= .9 for p in pairs for v in p["fraction_of_thinking"]),
                  content_review_sha256=sha(args.review), quote_locations=citations,
                  reviewer="Single Codex-assisted full-text review; not independent human annotation",
                  semantic_outcomes=dict(Counter(r["decision"] for r in review["records"])),
                  semantic_gate_decision=review["gate_decision"],
                  exact_checks=dict(fraction=str(Fraction(150, 37)), unequal_solution=[4, -2, -8],
                                    rescaled_coefficients=list(map(str, coefficients)), rescaled_sum=str(sum(coefficients))),
                  pool=dict(prior_plan_sha256=sha(plan_path), excluded_unique=len(forbidden),
                            eligible_under_latest_ledger=len(eligible), selected_new_questions=0,
                            limitation="No new split frozen; full repository-wide reservation reconciliation deferred after content stop"),
                  reviewed_keys=review_keys,
                  source_sha256=sha(Path(__file__)),
                  delivery_report_sha256=sha(delivery / "EasySteer_ReBalance_独立研究报告.md"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "semantic_outcomes", "pool", "capture_positions_at_least_90_percent")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
