"""Paper offline stage on saved greedy MATH-train continuations.

Token boundaries, rather than re-tokenized text, define completed reasoning
steps. Boundary tokens are excluded from probability aggregation. An unfinished
last step is excluded; completed preceding steps remain usable. No gold answers
or benchmark outputs are read. PCA/Ridge uses the author's step split and seed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import torch
from transformers import AutoTokenizer

from calibrate_own_vector import read, save


def prepare(source, out):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["temperature"] != 0 or manifest["count"] != 500:
        raise ValueError("Saved chosen-token probabilities require 500 greedy samples")
    if (out / "protocol.json").exists():
        raise FileExistsError("Paper calibration protocol already exists")
    model_path = manifest["model"]
    tok = AutoTokenizer.from_pretrained(model_path)
    boundary = {i for t, i in tok.get_vocab().items() if "ĊĊ" in t}
    end_id = tok.encode("</think>", add_special_tokens=False)
    assert len(end_id) == 1
    records = []
    positions = []
    for question, row in enumerate(read(source / "generations.jsonl")):
        assert row["train_index"] == manifest["train_indices"][question]
        ids, logps = row["token_ids"], row["logprobs"]
        assert len(ids) == len(logps)
        assert all(np.isfinite(p) and p <= 1e-5 for p in logps)
        closed = end_id[0] in ids
        end = ids.index(end_id[0]) if closed else len(ids)
        start = 0
        previous = None
        spans = []
        for stop in range(end + 1):
            delimiter = stop < end and ids[stop] in boundary
            if not delimiter and not (stop == end and closed):
                continue
            if stop > start:
                confidence = float(np.exp(np.mean(logps[start:stop])))
                variance = 0.0 if previous is None else (confidence-previous)**2/4
                records.append(dict(question=question, start=start, stop=stop,
                                    confidence=confidence, variance=variance))
                spans.append(start + len(row["prompt_token_ids"]))
                previous = confidence
            start = stop + 1
        positions.append(dict(question=question, positions=spans, think_stop=end))
    assert len(positions) == 500 and records
    save(out / "steps.json", records)
    save(out / "positions.json", positions)
    c = np.array([r["confidence"] for r in records])
    v = np.array([r["variance"] for r in records])
    qc, qv = np.quantile(c, [.25, .75]), np.quantile(v, [.25, .75])
    over = (c <= qc[0]) & (v >= qv[1])
    under = (c >= qc[1]) & (v <= qv[0])
    assert over.any() and under.any()
    save(out / "protocol.json", dict(model=model_path, source=str(source),
        source_sha256=hashlib.sha256((source / "generations.jsonl").read_bytes()).hexdigest(),
        questions=500, steps=len(records), confidence_quantiles=qc.tolist(),
        variance_quantiles=qv.tolist(), overthinking_steps=int(over.sum()),
        underthinking_steps=int(under.sum()), seed=42, pca_components=64,
        ridge_alpha=1.0, held_out_fraction=.2, split_unit="reasoning step (author script)",
        confidence="geometric mean of raw max probabilities; greedy chosen=max",
        steps_policy="Exclude delimiter token and incomplete trailing step",
        classification="paper Eq.5 joint confidence and variance; no lexicon"))


def collect(source, out):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    if (out / "layer_0.npy").exists():
        raise FileExistsError("Hidden arrays already exist; no implicit overwrite")
    rows = read(source / "generations.jsonl")
    positions = json.loads((out / "positions.json").read_text())
    count = sum(len(r["positions"]) for r in positions)
    protocol = json.loads((out / "protocol.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(protocol["model"],
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    layers = len(model.model.layers)
    width = model.config.hidden_size
    maps = [np.lib.format.open_memmap(out / f"layer_{i}.npy", mode="w+",
            dtype=np.float32, shape=(count, width)) for i in range(layers+1)]
    capture = {}
    selected = []
    def hook(index):
        def capture_layer(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            capture[index] = h[0, selected].detach().float().cpu().numpy()
        return capture_layer
    handles = [model.model.embed_tokens.register_forward_hook(hook(0))]
    handles += [layer.register_forward_hook(hook(i+1))
                for i, layer in enumerate(model.model.layers[:-1])]
    handles.append(model.model.norm.register_forward_hook(hook(layers)))
    offset = 0
    started = time.time()
    for i, (row, pos) in enumerate(zip(rows, positions, strict=True)):
        selected = pos["positions"]
        if not selected:
            continue
        ids = row["prompt_token_ids"] + row["token_ids"][:pos["think_stop"]]
        with torch.inference_mode():
            result = model.model(torch.tensor([ids], device="cuda"),
                use_cache=False, return_dict=True, output_hidden_states=(i == 0))
        if i == 0:
            assert len(result.hidden_states) == len(maps)
            for j in range(len(maps)):
                reference = result.hidden_states[j][0, selected].float().cpu().numpy()
                assert np.array_equal(reference, capture[j]), j
            save(out / "layer_hook_check.json", dict(layers=len(maps),
                 exact=True, checked_positions=len(selected)))
        for j in range(len(maps)):
            maps[j][offset:offset+len(selected)] = capture.pop(j)
        offset += len(selected)
        del result
        if (i+1) % 25 == 0:
            print(f"All-layer replay {i+1}/500, {time.time()-started:.1f}s", flush=True)
    assert offset == count
    for handle in handles:
        handle.remove()
    for array in maps:
        array.flush()
    save(out / "collection.json", dict(layers=layers+1, steps=count, width=width,
         seconds=time.time()-started, dtype="float32 (exact conversion from bfloat16)"))


def select(out):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_limits
    if (out / "selected_layer.json").exists():
        raise FileExistsError("Layer selection already completed")
    steps = json.loads((out / "steps.json").read_text())
    config = json.loads((out / "collection.json").read_text())
    y = np.array([r["confidence"] for r in steps], dtype=np.float32)
    train, valid = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    results = []
    started = time.time()
    with threadpool_limits(limits=8):
        for layer in range(config["layers"]):
            x = np.load(out / f"layer_{layer}.npy", mmap_mode="r")
            pca = PCA(n_components=64, random_state=42)
            xt = pca.fit_transform(x[train])
            xv = pca.transform(x[valid])
            reg = Ridge(alpha=1.0).fit(xt, y[train])
            r2 = float(r2_score(y[valid], reg.predict(xv)))
            results.append(dict(layer=layer, r2=r2, pca_solver=pca._fit_svd_solver))
            save(out / "layer_scores.json", results)
            print(f"Layer {layer}: R2={r2:.6f}, elapsed={time.time()-started:.1f}s", flush=True)
            del x, xt, xv, pca, reg
    best = max(results, key=lambda r: r["r2"])
    save(out / "selected_layer.json", dict(best=best, scores=results,
         seconds=time.time()-started,
         criterion="maximum held-out confidence R2; not benchmark accuracy"))


def vector(out):
    protocol = json.loads((out / "protocol.json").read_text())
    best = json.loads((out / "selected_layer.json").read_text())["best"]["layer"]
    steps = json.loads((out / "steps.json").read_text())
    c = np.array([r["confidence"] for r in steps])
    v = np.array([r["variance"] for r in steps])
    cl, ch = protocol["confidence_quantiles"]
    vl, vh = protocol["variance_quantiles"]
    over, under = (c <= cl) & (v >= vh), (c >= ch) & (v <= vl)
    x = np.load(out / f"layer_{best}.npy", mmap_mode="r")
    mu_o, mu_u = x[over].mean(axis=0, dtype=np.float64), x[under].mean(axis=0, dtype=np.float64)
    distance = np.linalg.norm(mu_o-mu_u)
    assert distance > 0
    direction = (mu_o-mu_u)/distance
    threshold = .5 * direction.dot(mu_o+mu_u)
    moderate = direction.dot(mu_o)-threshold
    aggressive = float(np.max(x[over] @ direction-threshold))
    assert np.isclose(moderate, distance/2)
    assert aggressive >= moderate > 0
    tensor = torch.from_numpy(direction.astype(np.float32))
    assert torch.isclose(tensor.norm(), torch.tensor(1.), atol=1e-6)
    torch.save(tensor, out / "paper_unit_vector.pt")
    save(out / "geometry.json", dict(layer=best, norm=tensor.norm().item(),
        prototype_distance=float(distance), moderate_overthinking=float(moderate),
        aggressive_overthinking=aggressive, projection_threshold=float(threshold),
        overthinking_steps=int(over.sum()), underthinking_steps=int(under.sum()),
        vector_sha256=hashlib.sha256((out / "paper_unit_vector.pt").read_bytes()).hexdigest(),
        control_status="geometry only; missing paper control constants not guessed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "collect", "select", "vector"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "collect"):
        globals()[args.stage](args.source, args.output)
    else:
        globals()[args.stage](args.output)
