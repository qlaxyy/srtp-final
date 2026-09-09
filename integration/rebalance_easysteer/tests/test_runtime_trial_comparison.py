"""Reject unfair speed comparisons; expose the first changed output token."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_7b_runtime import compare_results


def test_runtime_comparison():
    protocol = dict(model="7B", dataset="calibration64", max_tokens=16000,
                    max_model_len=17408, temperature=0, top_p=1, seed=42,
                    dynamic_params={"initial_coef": -1}, easysteer_output_layer=21,
                    chunked_prefill=True, max_num_batched_tokens=2048,
                    gpu_memory_utilization=.92, async_scheduling=False,
                    max_num_seqs=32, profiling_enabled=False)
    provenance = dict(dataset_sha256="data", vector_sha256="vector",
                      calibration_fit_sha256="fit", commit="same")
    first = dict(status="diagnostic_completed", protocol=protocol, provenance=provenance,
                 rebalance_dynamic=dict(summary={"generation_seconds": 200},
                     records=[dict(problem=str(i), token_ids=[10, 20, 30]) for i in range(64)]))
    second = deepcopy(first)
    second["protocol"]["max_num_seqs"] = 64
    second["rebalance_dynamic"]["summary"]["generation_seconds"] = 100
    result = compare_results(first, second)
    assert result["output_equal"] and result["generation_speed_ratio32_over64"] == 2
    second["rebalance_dynamic"]["records"][9]["token_ids"][1] = 99
    result = compare_results(first, second)
    assert result["differences"] == [dict(index=9, first_different_token=1, tokens32=3, tokens64=3)]
    for field, value in (("max_tokens", 8192), ("dynamic_params", {"initial_coef": -2}),
                         ("profiling_enabled", True)):
        changed = deepcopy(second)
        changed["protocol"][field] = value
        try:
            compare_results(first, changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Must reject incompatible {field}")


if __name__ == "__main__":
    test_runtime_comparison()
    print("PASS runtime comparison: cap/control/profile mismatch rejected; token divergence located")
