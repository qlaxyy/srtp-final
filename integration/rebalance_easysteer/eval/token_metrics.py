"""Output-length accounting: every answer, including truncated answers, counts."""
from math import ceil
from statistics import mean, median


def length_metrics(records, max_tokens):
    if max_tokens <= 0 or not records:
        raise ValueError("Positive generation limit and nonempty records required")
    counts = sorted(record["tokens"] for record in records)
    if any(n < 0 or n > max_tokens for n in counts):
        raise ValueError("Recorded length exceeds the declared generation budget")
    capped = sum(record["finish_reason"] == "length" and
                 record["tokens"] == max_tokens for record in records)
    return dict(
        mean_tokens=mean(counts), total_tokens=sum(counts), capped=capped,
        capped_rate=capped / len(records),
        reached_token_limit=sum(n == max_tokens for n in counts),
        median_tokens=median(counts),
        p95_tokens=counts[ceil(.95 * len(counts)) - 1],
        max_observed_tokens=counts[-1],
        length_policy="all generated tokens; capped and incorrect answers included",
    )
