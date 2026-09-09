"""No GPU required: distinguish ordinary recomputation from lost steering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from runtime_guards import guard_dynamic_preemption, generate_with_checkpoint


def test_preemption_guard():
    handled = []
    scheduler = SimpleNamespace(_preempt_request=lambda r, t: handled.append(r))
    counts = guard_dynamic_preemption(scheduler)
    plain = SimpleNamespace(steer_vector_request=None, num_output_tokens=20)
    fresh = SimpleNamespace(steer_vector_request=SimpleNamespace(algorithm="rebalance"),
                            num_output_tokens=0)
    dynamic = SimpleNamespace(steer_vector_request=fresh.steer_vector_request,
                              num_output_tokens=20, request_id="r1")
    scheduler._preempt_request(plain, 1)
    scheduler._preempt_request(fresh, 2)
    try:
        scheduler._preempt_request(dynamic, 3)
    except RuntimeError as exc:
        assert "generated_tokens=20" in str(exc)
    else:
        raise AssertionError("Must reject lost dynamic history")
    assert handled == [plain, fresh]
    assert counts == {"events": 3, "rejected_dynamic_events": 1}
    scheduler = SimpleNamespace(_preempt_request=lambda r, t: handled.append(r))
    restored = guard_dynamic_preemption(
        scheduler, SimpleNamespace(supports_kv_replay=True))
    scheduler._preempt_request(dynamic, 4)
    assert restored == {"events": 1, "rejected_dynamic_events": 0}
    assert handled[-1] is dynamic


def test_evaluation_checkpoint():
    class FakeLLM:
        def __init__(self, fail=False):
            self.llm_engine = self
            self.fail, self.turn = fail, 0

        def enqueue(self, prompts, sampling_params, steering):
            assert prompts == ["a", "b"] and steering == "chosen-vector"
            self.output_processor = SimpleNamespace(request_states={
                "a-internal": SimpleNamespace(external_req_id="a"),
                "b-internal": SimpleNamespace(external_req_id="b"),
            })
            return ["a-internal", "b-internal"]

        def has_unfinished_requests(self):
            return self.turn < 2

        def step(self):
            self.turn += 1
            if self.fail and self.turn == 2:
                raise RuntimeError("simulated interruption")
            rid = "b" if self.turn == 1 else "a"
            return [SimpleNamespace(finished=True, request_id=rid,
                prompt_token_ids=[10], outputs=[SimpleNamespace(
                    token_ids=[11], text=rid, finish_reason="stop")])]

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "complete.jsonl"
        results = generate_with_checkpoint(FakeLLM(), ["a", "b"], None,
                                           "chosen-vector", path)
        assert [r.request_id for r in results] == ["a", "b"]
        assert [json.loads(s)["local_index"] for s in path.read_text().splitlines()] == [1, 0]
        partial = Path(folder) / "interrupted.jsonl"
        try:
            generate_with_checkpoint(FakeLLM(True), ["a", "b"], None,
                                      "chosen-vector", partial)
        except RuntimeError as exc:
            assert str(exc) == "simulated interruption"
        else:
            raise AssertionError("Expected interruption")
        saved = json.loads(partial.read_text())
        assert saved["local_index"] == 1 and saved["text"] == "b"


def test_resume_only_missing_answers():
    class FakeLLM:
        def __init__(self):
            self.llm_engine = self
            self.done = False

        def enqueue(self, prompts, sampling_params, steering):
            assert prompts == ["a", "c"], "Retained question b must never be generated again"
            self.output_processor = SimpleNamespace(request_states={
                "r0": SimpleNamespace(external_req_id="external-a"),
                "r1": SimpleNamespace(external_req_id="external-c")})
            return ["r0", "r1"]

        def has_unfinished_requests(self):
            return not self.done

        def step(self):
            self.done = True
            return [SimpleNamespace(finished=True, request_id="external-" + s,
                prompt_token_ids=[10], outputs=[SimpleNamespace(token_ids=[11],
                text=s, finish_reason="stop")]) for s in ("c", "a")]

    with tempfile.TemporaryDirectory() as folder:
        old, new = Path(folder) / "old.jsonl", Path(folder) / "new.jsonl"
        row = dict(local_index=1, prompt_token_ids=[10], token_ids=[777],
                   text="retained-b", finish_reason="stop")
        old.write_text(json.dumps(row) + "\n")
        original = old.read_bytes()
        result = generate_with_checkpoint(FakeLLM(), ["a", "b", "c"],
            SimpleNamespace(max_tokens=16000), None, new, resume_path=old)
        assert [r.outputs[0].text for r in result] == ["a", "retained-b", "c"]
        assert result[1].outputs[0].token_ids == [777] and old.read_bytes() == original
        assert sorted(json.loads(s)["local_index"] for s in new.read_text().splitlines()) == [0, 1, 2]
        old.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        try:
            generate_with_checkpoint(FakeLLM(), ["a", "b", "c"],
                SimpleNamespace(max_tokens=16000), None, Path(folder) / "bad.jsonl", resume_path=old)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate saved answer was accepted")


if __name__ == "__main__":
    test_preemption_guard()
    test_evaluation_checkpoint()
    test_resume_only_missing_answers()
    print("PASS dynamic preemption guard and evaluation checkpoint")
