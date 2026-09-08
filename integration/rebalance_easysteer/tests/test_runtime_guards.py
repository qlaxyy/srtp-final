"""No GPU required: distinguish ordinary recomputation from lost steering."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from runtime_guards import guard_dynamic_preemption


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


if __name__ == "__main__":
    test_preemption_guard()
    print("PASS dynamic preemption guard")
