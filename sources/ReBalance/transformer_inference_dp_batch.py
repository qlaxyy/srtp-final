"""Single-GPU batched generation for ReBalance steering-vector extraction.

This is an opt-in throughput experiment.  It reuses the validated fast-path
helpers from ``transformer_inference_dp_fast.py`` and only batches the greedy
generation stage.  Hidden-state extraction remains per sample so its saved
format and numerical values match the reference pipeline.
"""

import argparse
import json
import os
from re import split as rsplit

import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

try:
    import transformer_inference_dp_fast as base
except ImportError:  # Local development copy uses the optimized source name.
    import transformer_inference_dp as base


class BatchGreedyTokenLogprobRecorder(LogitsProcessor):
    """Keep one selected-token log probability per batch row and step."""

    def __init__(self):
        self._steps = []

    def __call__(self, input_ids, scores):
        values = torch.amax(scores, dim=-1) - torch.logsumexp(scores, dim=-1)
        self._steps.append(values.detach().to(torch.float32).clone())
        return scores

    def as_cpu_matrix(self):
        if not self._steps:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(self._steps, dim=0).transpose(0, 1).cpu()


def _trim_at_eos(token_ids, eos_token_id):
    """Drop batch-padding tokens after the first generated EOS, retaining EOS."""
    ids = token_ids.detach().cpu()
    if eos_token_id is None:
        return ids
    positions = (ids == eos_token_id).nonzero(as_tuple=False)
    if positions.numel():
        ids = ids[: int(positions[0].item()) + 1]
    return ids


def _segment_confidences(tokenizer, response_text, gen_logps):
    text_before_think = response_text.split("</think>")[0]
    segments = rsplit(r"\n\n+", text_before_think)
    values = []
    start = 0
    for segment in segments:
        seg_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        end = start + len(seg_ids)
        if gen_logps.numel() and end > start and end <= gen_logps.numel():
            values.append(base._summarize_selected_logprobs(gen_logps[start:end], policy="avg2"))
        elif end > start:
            values.append(0.0)
        start = end
    return values


def worker(args):
    base.set_seeds(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "test.jsonl")
    questions = base.read_jsonl(dataset_path)
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"origin_temp{args.temperature}_maxlen{args.max_generated_tokens}"
    shard_file = os.path.join(output_dir, f"{base_name}.shard0.jsonl")

    existing_idx, existing_q, num_lines = base.load_existing_indices(shard_file)
    existing_map = base._read_jsonl_map_by_idx(shard_file)
    model, tokenizer = base.load_model_and_tokenizer_single_gpu(args, device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    print(f"[batch] shard_file = {shard_file}")
    print(f"[batch] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}, lines={num_lines}")
    print(f"[batch] batch_size={args.batch_size}, total={len(questions)}")
    pbar = tqdm(total=len(questions), desc="Batched extraction")

    # Rescue missing hidden files for already persisted responses.
    pending = []
    for i, q in enumerate(questions):
        hidden_path = os.path.join(output_dir, f"hidden_{i}.pt")
        if i in existing_idx or q.get("problem") in existing_q:
            if not os.path.exists(hidden_path):
                entry = existing_map.get(i)
                if entry and entry.get("generated_responses"):
                    response = entry["generated_responses"][0]
                    full_text = base._reconstruct_full_text(
                        tokenizer, sys_prompt, q.get("problem", ""), response
                    )
                    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
                    base.save_qwen2_think_split_tokens_only(
                        model, tokenizer, full_ids, full_text, hidden_path,
                        hs_device=args.hs_device, skip_lm_head=True,
                    )
                    del full_ids
                    base._clear_cuda()
            pbar.update(1)
        else:
            pending.append(i)

    def run_batch(indices):
        prompts = []
        for i in indices:
            q = questions[i]
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q["problem"]},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        recorder = BatchGreedyTokenLogprobRecorder()
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generated_tokens,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=LogitsProcessorList([recorder]),
                )
        except torch.cuda.OutOfMemoryError:
            del inputs
            base._clear_cuda()
            if len(indices) == 1:
                print(f"[OOM] generation idx={indices[0]}; skipped")
                pbar.update(1)
                return
            middle = len(indices) // 2
            print(f"[WARN] batch OOM; retrying {len(indices)} as smaller batches")
            run_batch(indices[:middle])
            run_batch(indices[middle:])
            return

        prompt_width = inputs["input_ids"].shape[1]
        logp_matrix = recorder.as_cpu_matrix()

        # Hidden-state dumping remains single-sample and therefore preserves the
        # original artifact layout and avoids a large padded all-layer tensor.
        for row_pos, i in enumerate(indices):
            q = questions[i]
            gen_ids = _trim_at_eos(
                output.sequences[row_pos, prompt_width:], tokenizer.eos_token_id
            )
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            full_text = prompts[row_pos] + response
            gen_logps = logp_matrix[row_pos, : gen_ids.numel()]
            confidences = _segment_confidences(tokenizer, response, gen_logps)
            hidden_path = os.path.join(output_dir, f"hidden_{i}.pt")

            try:
                full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
                base.save_qwen2_think_split_tokens_only(
                    model, tokenizer, full_ids, full_text, hidden_path,
                    hs_device=args.hs_device, skip_lm_head=True,
                )
                del full_ids
            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] hidden save OOM idx={i}; response retained for rescue")
                base._clear_cuda()
            except Exception as exc:
                print(f"[WARN] hidden save failed idx={i}: {exc}")

            entry = {
                "idx": i,
                "question": q.get("problem", ""),
                "generated_responses": [response],
                "gold_answer": q.get("answer", ""),
                "sentence_confidences": confidences,
            }
            with open(shard_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            pbar.update(1)

        del output, inputs, logp_matrix
        base._clear_cuda()

    for start in range(0, len(pending), args.batch_size):
        run_batch(pending[start : start + args.batch_size])

    pbar.close()
    print(f"[batch] Done: {shard_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_generated_tokens", type=int, default=512)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--hs_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--score_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    if args.num_gpus != 1:
        raise ValueError("This experimental batched runner currently supports one GPU.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    worker(args)


if __name__ == "__main__":
    main()
