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


if __name__ == "__main__":
    test_preemption_guard()
    test_evaluation_checkpoint()
    print("PASS dynamic preemption guard and evaluation checkpoint")
