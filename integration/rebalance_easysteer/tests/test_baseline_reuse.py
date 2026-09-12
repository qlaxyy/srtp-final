"""CPU regressions for invalid controls previously accepted by the entry point."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))
from baseline_reuse import execution_identity, validate_saved_baseline, REQUIRED_SOURCES


def fixture():
    protocol = dict(model="model", dataset="dataset", offset=0, limit=2, max_tokens=4,
                    max_model_len=128, temperature=.7, top_p=.95, seed=42,
                    execution_mode="in_graph", max_num_seqs=128, gpu_memory_utilization=.9,
                    async_scheduling=True, chunked_prefill=False, max_num_batched_tokens=128,
                    easysteer_output_layer=20, steering_algorithm="rebalance",
                    dynamic_params={"initial_coef": -1}, run_order=["baseline", "rebalance_dynamic"],
                    profiling_enabled=False, diagnostic_only=False)
    environment = {key: "same" for key in (
        "python", "torch", "cuda", "vllm", "gpu", "transformers", "tokenizers",
        "triton", "cuda_driver", "numeric_environment",
    )}
    provenance = dict(commit="new-docs-commit", git_status="", dataset_sha256="dataset",
                      vector_sha256="vector", execution_identity={
                          "version": "inference-content-v1", "source_sha256": {"code": "source"},
                          "model_files_sha256": {"weights": "model-content"},
                      })
    current = dict(protocol=protocol, environment=environment, provenance=provenance)
    records = [dict(dataset_index=0, problem="p0", gold="2", tokens=4,
                    token_ids=[10, 11, 2, 99], correct=True, finish_reason="stop",
                    thinking_tokens=2, thinking_ended=True, answer_tokens=1),
               dict(dataset_index=1, problem="p1", gold="4", tokens=4,
                    token_ids=[5, 6, 7, 8], correct=False, finish_reason="length",
                    thinking_tokens=4, thinking_ended=False, answer_tokens=0)]
    summary = dict(examples=2, correct=1, accuracy=.5, mean_tokens=4, total_tokens=8,
                   capped=1, capped_rate=.5, reached_token_limit=2, median_tokens=4,
                   p95_tokens=4, max_observed_tokens=4,
                   length_policy="all generated tokens; capped and incorrect answers included",
                   mean_thinking_tokens=3, mean_answer_tokens=.5, thinking_not_ended=1,
                   generation_seconds=2.)
    saved = copy.deepcopy(current)
    saved.update(status="completed", baseline=dict(records=records, summary=summary))
    saved["provenance"]["commit"] = "older-docs-commit"
    examples = [dict(problem="p0", answer="2"), dict(problem="p1", answer="4")]
    return saved, current, examples


class ReuseValidation(unittest.TestCase):
    def test_compatible_content_accepts_docs_only_commit_and_retains_wrong_capped_row(self):
        saved, current, examples = fixture()
        group = validate_saved_baseline(saved, current, examples, 2)
        self.assertIs(group, saved["baseline"])
        self.assertEqual(group["summary"]["capped"], 1)
        self.assertEqual(len(group["records"]), 2)

    def test_historical_counterexamples_fail(self):
        mutations = {
            "source changed": lambda s: s["provenance"]["execution_identity"]["source_sha256"].update(code="different"),
            "model changed": lambda s: s["provenance"]["execution_identity"]["model_files_sha256"].update(weights="different"),
            "partial pair": lambda s: s.update(status="incomplete"),
            "wrong gold": lambda s: s["baseline"]["records"][0].update(gold="wrong"),
            "invalid count": lambda s: s["baseline"]["records"][0].update(tokens=17000),
            "dataset changed": lambda s: s["provenance"].update(dataset_sha256="different"),
            "sampling changed": lambda s: s["protocol"].update(temperature=.8),
            "legacy identity absent": lambda s: s["provenance"].pop("execution_identity"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                saved, current, examples = fixture(); mutate(saved)
                with self.assertRaises(ValueError):
                    validate_saved_baseline(saved, current, examples, 2)

    def test_incomplete_or_inconsistent_answer_and_summary_fail(self):
        mutations = {
            "omit wrong answer": lambda s: s["baseline"]["records"].pop(),
            "unfinished": lambda s: s["baseline"]["records"][0].update(finish_reason=None),
            "bad index": lambda s: s["baseline"]["records"][1].update(dataset_index=0),
            "bool token": lambda s: s["baseline"]["records"][0]["token_ids"].__setitem__(0, True),
            "wrong think length": lambda s: s["baseline"]["records"][0].update(thinking_tokens=3),
            "wrong answer length": lambda s: s["baseline"]["records"][0].update(answer_tokens=2),
            "string correct": lambda s: s["baseline"]["records"][0].update(correct="true"),
            "missing stop status": lambda s: s["baseline"]["records"][0].pop("thinking_ended"),
            "short truncation": lambda s: s["baseline"]["records"][1].update(token_ids=[5, 6], tokens=2),
            "wrong total": lambda s: s["baseline"]["summary"].update(total_tokens=4),
            "wrong mean": lambda s: s["baseline"]["summary"].update(mean_tokens=3.9),
            "wrong caps": lambda s: s["baseline"]["summary"].update(capped=2),
            "nonfinite timing": lambda s: s["baseline"]["summary"].update(generation_seconds=float("nan")),
            "zero timing": lambda s: s["baseline"]["summary"].update(generation_seconds=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                saved, current, examples = fixture(); mutate(saved)
                with self.assertRaises(ValueError):
                    validate_saved_baseline(saved, current, examples, 2)

    def test_graph_environment_assets_and_dirty_source_fail(self):
        for section, key, value in [
            ("protocol", "steering_algorithm", "other-graph"),
            ("protocol", "run_order", ["rebalance_dynamic", "baseline"]),
            ("protocol", "async_scheduling", False),
            ("protocol", "dynamic_params", {}),
            ("protocol", "profiling_enabled", True),
            ("protocol", "diagnostic_only", True),
            ("environment", "cuda_driver", "different-driver"),
            ("environment", "triton", "different-compiler"),
            ("environment", "numeric_environment", "different-cast-mode"),
            ("provenance", "vector_sha256", "different-vector"),
            ("provenance", "git_status", " M runtime.py"),
        ]:
            with self.subTest(key=key):
                saved, current, examples = fixture(); saved[section][key] = value
                with self.assertRaises(ValueError):
                    validate_saved_baseline(saved, current, examples, 2)

    def test_real_entry_calls_validator_before_model_and_records_reuse_hash(self):
        path = EVAL / "rebalance_dynamic_eval.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        block = next(node for node in main.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "args.baseline_result")
        model_call = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == "LLM")
        self.assertLess(block.end_lineno, model_call.lineno)
        executable = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), str(path), "exec")
        with TemporaryDirectory() as directory:
            file = Path(directory) / "saved.json"
            for valid in (True, False):
                saved, current, examples = fixture()
                if not valid: saved["baseline"]["records"][0]["gold"] = "wrong"
                file.write_text(json.dumps(saved), encoding="utf-8")
                env = dict(args=SimpleNamespace(baseline_result=file), json=json, hashlib=hashlib,
                           result=current, examples=examples, think_end_id=2,
                           validate_saved_baseline=validate_saved_baseline)
                if valid:
                    exec(executable, env)
                    self.assertTrue(current["provenance"]["reused_baseline"]["timing_is_historical"])
                else:
                    with self.assertRaises(ValueError): exec(executable, env)

    def test_identity_reads_same_size_same_mtime_weight_changes_and_new_assets(self):
        with TemporaryDirectory() as directory:
            project = Path(directory); model = project / "model"; model.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            for name in REQUIRED_SOURCES:
                path = project / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x = 1\r\n")
            (project / "README.md").write_text("docs")
            subprocess.run(["git", "add", "."], cwd=project, check=True, capture_output=True)
            (model / "config.json").write_text("{}")
            (model / "tokenizer.json").write_text("{}")
            weights = model / "model.safetensors"; weights.write_bytes(b"old!")
            initial = execution_identity(project, model)
            stat = weights.stat(); weights.write_bytes(b"new!")
            os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            changed = execution_identity(project, model)
            self.assertNotEqual(initial["model_files_sha256"], changed["model_files_sha256"])
            self.assertEqual(initial["source_sha256"], changed["source_sha256"])
            (model / "generation_config.json").write_text("{}")
            self.assertIn("generation_config.json", execution_identity(project, model)["model_files_sha256"])
            (project / "README.md").write_text("new docs")
            source = project / REQUIRED_SOURCES[0]; source.write_bytes(b"x = 1\n")
            self.assertEqual(initial["source_sha256"], execution_identity(project, model)["source_sha256"])
            source.write_bytes(b"x = 2\n")
            self.assertNotEqual(initial["source_sha256"], execution_identity(project, model)["source_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
