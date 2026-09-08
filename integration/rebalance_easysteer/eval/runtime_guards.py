"""Fail closed when KV recomputation would invalidate dynamic steering."""


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
