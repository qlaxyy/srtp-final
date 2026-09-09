"""CPU checks for fair length accounting and bounded, opt-in tracing."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from token_metrics import length_metrics
from decode_profiler import EngineStepProfiler


def test_capped_and_incorrect_answers_stay_in_denominator():
    rows = [dict(tokens=100, finish_reason="stop", correct=True),
            dict(tokens=4096, finish_reason="length", correct=False),
            dict(tokens=300, finish_reason="stop", correct=False)]
    result = length_metrics(rows, 4096)
    assert result["mean_tokens"] == (100 + 4096 + 300) / 3
    assert result["capped_rate"] == 1 / 3
    assert result["median_tokens"] == 300
    assert result["p95_tokens"] == result["max_observed_tokens"] == 4096
    # EOS exactly at the limit is not a forced truncation.
    rows[1]["finish_reason"] = "stop"
    result = length_metrics(rows, 4096)
    assert result["capped"] == 0 and result["reached_token_limit"] == 1
    try:
        length_metrics(rows, 2048)
    except ValueError:
        pass
    else:
        raise AssertionError("Must reject a mismatched generation budget")


def test_profiler_only_records_selected_steps_and_closes_once():
    events = []
    class FakeProfiler:
        def start(self): events.append("start")
        def step(self): events.append("step")
        def stop(self): events.append("stop")
        def export_chrome_trace(self, path):
            events.append("export")
            Path(path).write_text(json.dumps(events))

    with TemporaryDirectory() as folder:
        path = Path(folder) / "trace.json"
        profiler = EngineStepProfiler(path, start_step=2, steps=3, factory=FakeProfiler)
        outputs = [profiler.run_step(lambda i=i: i) for i in range(8)]
        profiler.close()
        assert outputs == list(range(8))
        assert events == ["start", "step", "step", "step", "stop", "export"]
        assert json.loads(path.read_text()) == events
        events.clear()
        # Interrupted engine steps still release the active profiler.
        profiler = EngineStepProfiler(Path(folder) / "partial.json", 0, 3, FakeProfiler)
        try:
            profiler.run_step(lambda: 1 / 0)
        except ZeroDivisionError:
            profiler.close()
        assert events == ["start", "stop", "export"]


def test_dataset_budgets_reach_paired_evaluator():
    bash = shutil.which("bash")
    if not bash and os.name == "nt":
        bash = str(Path(shutil.which("git")).parents[1] / "bin/bash.exe")
    if not bash or not Path(bash).is_file():
        raise RuntimeError("Bash required to validate the evaluation entrypoint")
    root = Path(__file__).resolve().parents[3]
    script = root / "integration/rebalance_easysteer/scripts/run_auto_baseline.sh"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        stub = tmp / "python-stub"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n',
                        encoding="utf-8", newline="\n")
        stub.chmod(0o755)
        source = script.read_text(encoding="utf-8")
        for name, value in (("VLLM", "easysteer-vllm026"), ("LEGACY", "rebalance")):
            source = source.replace(f"{name}=/root/autodl-tmp/venvs/{value}/bin/python",
                                    f"{name}='{stub.as_posix()}'")
        copied = tmp / "run.sh"
        copied.write_text(source, encoding="utf-8", newline="\n")
        for dataset, cap, count in (("gsm8k", 16000, 1319), ("math500", 16000, 500)):
            capture = tmp / f"{dataset}.args"
            env = {k:v for k,v in os.environ.items() if not k.startswith(("EVAL_", "BASELINE_"))}
            env.update(PROJECT_ROOT=root.as_posix(), MODEL="model", CALIBRATION_SOURCE="calibration",
                       OUTPUT=(tmp / dataset).as_posix(), CAPTURE=capture.as_posix(),
                       EVAL_ARTIFACTS_DIR=(tmp / "frozen").as_posix())
            subprocess.run([bash, copied.as_posix(), "--group", dataset], env=env,
                           check=True, capture_output=True)
            args = capture.read_text(encoding="utf-8").splitlines()
            for flag, value in (("--max-tokens", cap), ("--limit", count),
                                ("--max-model-len", 17408), ("--max-num-seqs", 32)):
                assert args[args.index(flag)+1] == str(value)
            assert "--chunked-prefill" in args
            assert args[args.index("--vector")+1].endswith("/frozen/auto_vector.pt")


if __name__ == "__main__":
    test_capped_and_incorrect_answers_stay_in_denominator()
    test_profiler_only_records_selected_steps_and_closes_once()
    test_dataset_budgets_reach_paired_evaluator()
    print("PASS accounting, bounded profiling, cleanup, and GSM8K/MATH command routing (no GPU)")
