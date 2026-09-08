"""Fail closed when KV recomputation would invalidate dynamic steering."""
import json


def guard_dynamic_preemption(scheduler):
    original = scheduler._preempt_request
    counts = {"events": 0, "rejected_dynamic_events": 0}

    def preempt(request, timestamp):
        counts["events"] += 1
        steering = getattr(request, "steer_vector_request", None)
        if (steering is not None and steering.algorithm == "rebalance"
                and request.num_output_tokens > 0):
            counts["rejected_dynamic_events"] += 1
            raise RuntimeError(
                "Dynamic ReBalance cannot yet replay steering through KV-cache "
                f"preemption (request={request.request_id}, "
                f"generated_tokens={request.num_output_tokens}). "
                "Stopped before generating an invalid comparison."
            )
        return original(request, timestamp)

    scheduler._preempt_request = preempt
    return counts


def generate_with_checkpoint(llm, prompts, params, steering, path):
    """Use the same engine loop as generate, retaining completed raw answers."""
    if path.exists():
        raise FileExistsError(path)
    request_ids = llm.enqueue(prompts, sampling_params=params, steering=steering)
    states = llm.llm_engine.output_processor.request_states
    indices = {states[rid].external_req_id: i for i, rid in enumerate(request_ids)}
    assert len(indices) == len(prompts)
    completed = {}
    with path.open("x", encoding="utf-8") as stream:
        while llm.llm_engine.has_unfinished_requests():
            for result in llm.llm_engine.step():
                if not result.finished:
                    continue
                index = indices[result.request_id]
                assert index not in completed and len(result.outputs) == 1
                completed[index] = result
                output = result.outputs[0]
                stream.write(json.dumps(dict(
                    local_index=index, prompt_token_ids=result.prompt_token_ids,
                    token_ids=list(output.token_ids), text=output.text,
                    finish_reason=output.finish_reason,
                ), ensure_ascii=False) + "\n")
                stream.flush()
                if len(completed) % 10 == 0:
                    print(f"Saved {len(completed)}/{len(prompts)} answers", flush=True)
    assert len(completed) == len(prompts)
    return [completed[i] for i in range(len(prompts))]
