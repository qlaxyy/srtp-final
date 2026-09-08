"""One paired calibration prompt: uninterrupted vs two forced KV evictions."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]

import torch
from easysteer.vectors import from_pt_direction
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec
from rebalance_static_eval import build_prompt
from runtime_guards import generate_with_checkpoint, guard_dynamic_preemption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((args.calibration / "fit.json").read_text())
    vector = args.calibration / "auto_vector.pt"
    assert hashlib.sha256(vector.read_bytes()).hexdigest() == fit["vector_sha256"]
    row = json.loads((args.calibration / "calibration/generations.jsonl").open().readline())
    tokenizer = AutoTokenizer.from_pretrained(fit["model"], local_files_only=True)
    prompt = build_prompt(tokenizer, row["problem"])
    boundaries = sorted(i for token, i in tokenizer.get_vocab().items() if "ĊĊ" in token)
    params = dict(fit["parameters"], boundary_token_ids=boundaries,
                  think_start_token_id=tokenizer.encode("<think>", add_special_tokens=False)[0],
                  think_end_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0])
    spec = SteeringSpec(vectors=[VectorSpec(
        name="replay_check", data=from_pt_direction(str(vector), layers=[fit["decoder_output_layer"]]),
        layers=[fit["decoder_output_layer"]], normalize=False, algorithm="rebalance",
        apply=ApplySpec(generation_tokens=boundaries), params=params,
    )])
    started = time.perf_counter()
    llm = LLM(model=fit["model"], dtype="bfloat16", max_model_len=2048,
              max_num_seqs=4, max_num_batched_tokens=128,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              enable_steer_vector=True, steer_algorithms=["rebalance"],
              steer_graph_mode="in_graph", seed=42)
    core = llm.llm_engine.engine_core.engine_core
    state = core.model_executor.driver_worker.worker.model_runner.steer_vector_state
    counts = guard_dynamic_preemption(core.scheduler, state)
    sampling = SamplingParams(temperature=0., max_tokens=384, min_tokens=384,
                              ignore_eos=True, seed=42)
    original_scales = state.token_scales

    def old_scales(batch):
        return torch.repeat_interleave(state.batch_scales(batch),
                                       torch.diff(batch.query_start_loc[:batch.num_reqs + 1]))

    # The reference uses the previous uninterrupted scale expansion.
    state.token_scales = old_scales
    before = generate_with_checkpoint(llm, [prompt], sampling, spec,
                                      args.output / "uninterrupted.jsonl")[0]
    state.token_scales = original_scales
    scheduled = core.scheduler.schedule
    forced = []

    def force_eviction():
        if len(forced) < 2:
            target = [64, 192][len(forced)]
            for request in list(core.scheduler.running):
                if request.num_output_tokens >= target:
                    core.scheduler.running.remove(request)
                    forced.append(request.num_output_tokens)
                    core.scheduler._preempt_request(request, time.monotonic())
                    break
        return scheduled()

    core.scheduler.schedule = force_eviction
    after = generate_with_checkpoint(llm, [prompt], sampling, spec,
                                     args.output / "forced_replay.jsonl")[0]
    a, b = list(before.outputs[0].token_ids), list(after.outputs[0].token_ids)
    result = dict(
        purpose="7B calibration prompt, old uninterrupted scales vs historical replay",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        vector_sha256=fit["vector_sha256"], calibration_train_index=row["train_index"],
        forced_at_output_tokens=forced, replay_counts=state.replay_counts,
        preemptions=counts, tokens=[len(a), len(b)], exact_token_match=a == b,
        boundary_tokens_in_reference=sum(t in boundaries for t in a),
        first_difference=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None),
        seconds_including_startup=time.perf_counter()-started,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    assert forced == [64, 192] and state.replay_counts == {"suspended": 2, "restored": 2}
    assert result["boundary_tokens_in_reference"] > 0
    assert a == b, "Replayed run differs; inspect before formal evaluation"


if __name__ == "__main__":
    main()
