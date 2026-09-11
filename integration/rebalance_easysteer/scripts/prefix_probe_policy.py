"""Causal checkpoint and strict answer-consistency helpers (CPU only)."""
import re


def checkpoints(token_ids, boundaries, end_id, spacing=1024):
    """Return prefix lengths ending at a generated delimiter, before </think>."""
    result = []
    previous = 0
    for index, token in enumerate(token_ids):
        if token == end_id:
            break
        length = index + 1
        if token in boundaries and length - previous >= spacing:
            result.append(length)
            previous = length
    return result


def boxed_key(completion):
    """Read the first complete forced box; never fall back to a last number.

    The prompt ends in \\boxed{. Only conservative textual normalization is
    used for the stopping decision; correctness is independently author graded.
    """
    depth = 1
    for index, char in enumerate(completion):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            answer = completion[:index].strip()
            answer = answer.replace(r"\tfrac", r"\frac")
            answer = answer.replace(r"\dfrac", r"\frac")
            answer = answer.replace(r"\left", "").replace(r"\right", "")
            return re.sub(r"\s+", "", answer)
    return ""


def first_consistent(keys, consecutive=3):
    previous = ""
    run = 0
    for index, key in enumerate(keys):
        run = run + 1 if key and key == previous else int(bool(key))
        previous = key
        if run >= consecutive:
            return index
    return None


def account(original_total, original_thinking, points, output_lengths,
            keys, prompt_lengths, consecutive=3):
    """Account a hypothetical saved-path policy, not actual online savings."""
    assert len(points) == len(output_lengths) == len(keys) == len(prompt_lengths)
    chosen = first_consistent(keys, consecutive)
    used = len(points) if chosen is None else chosen + 1
    trial_tokens = sum(output_lengths[:used])
    reasoning = original_thinking if chosen is None else points[chosen]
    generated = original_total + trial_tokens if chosen is None else reasoning + trial_tokens
    return dict(stop_index=chosen, used_probes=used,
                hypothetical_thinking_tokens=reasoning,
                hypothetical_generated_tokens=generated,
                probe_output_tokens=trial_tokens,
                probe_prefill_input_tokens=sum(prompt_lengths[:used]),
                original_total_tokens=original_total,
                original_thinking_tokens=original_thinking)


def test_policy():
    # Causal grid uses only observed delimiters; a future suffix cannot move it.
    seq = [1, 1, 2, 1, 2, 1, 1, 2, 9, 2]
    assert checkpoints(seq, {2}, 9, 3) == [3, 8]
    assert checkpoints(seq[:8], {2}, 9, 3) == [3, 8]
    assert first_consistent([""] * 12) is None
    assert first_consistent(["a", "a", "", "a", "a", "a"]) == 5
    assert first_consistent(["a", "b", "b", "b", "a"]) == 3
    assert boxed_key(r"\frac{1}{2}} more text") == r"\frac{1}{2}"
    assert boxed_key(r"\frac{1}{2}") == ""
    assert boxed_key("}") == ""
    assert boxed_key(r"\tfrac{1}{ 2}}") == r"\frac{1}{2}"
    cost = account(10000, 9500, [1024, 2050, 3100, 4200],
                   [10, 20, 30, 40], ["a", "a", "a", "b"],
                   [1100, 2126, 3176, 4276])
    assert cost["hypothetical_generated_tokens"] == 3160
    assert cost["probe_prefill_input_tokens"] == 6402
    cost = account(10000, 9500, [1024], [128], [""], [1100])
    assert cost["hypothetical_generated_tokens"] == 10128
    return 11


if __name__ == "__main__":
    print("Passed", test_policy(), "causal/empty-answer/cost checks; model calls0")
