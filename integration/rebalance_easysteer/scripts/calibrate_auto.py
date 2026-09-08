"""Automatic code-based calibration v2 for Qwen/Llama-style reasoning decoders.

All choices are made on MATH-train calibration data. This is a new adaptation,
not an exact reproduction: arithmetic confidence is aligned offline/online,
the LDA midpoint sign is corrected, and the third curve target is made feasible.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np
import torch

from calibrate_own_vector import ROOT, generate, read, save


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(args):
    from transformers import AutoTokenizer
    out, source = args.output, args.source
    if (out / "protocol.json").exists():
        raise FileExistsError("Calibration already prepared")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["temperature"] != 0 or manifest["count"] != 500:
        raise ValueError("Expected 500 saved greedy calibration samples")
    if Path(manifest["model"]).resolve() != args.model.resolve():
        raise ValueError("Calibration generations belong to another model")
    split = json.loads((source / "split_check.json").read_text())
    if any(split[n]["exact_normalized_problem_overlap"] != 0
           for n in ("Math_Math500", "Math_GSM8K")):
        raise ValueError("Calibration overlaps evaluation")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    boundary = {i for t, i in tok.get_vocab().items() if "ĊĊ" in t}
    end = tok.encode("</think>", add_special_tokens=False)
    start_id = tok.encode("<think>", add_special_tokens=False)
    if not boundary or len(end) != 1 or len(start_id) != 1:
        raise ValueError("This tokenizer needs an explicit reasoning-boundary adapter")
    author = load_module("auto_author_labels", ROOT / "sources/ReBalance/hidden_analysis_auto.py")
    records, positions = [], []
    rows = read(source / "generations.jsonl")
    if len(rows) != 500:
        raise ValueError("Incomplete calibration generation")
    for q, row in enumerate(rows):
        assert row["train_index"] == manifest["train_indices"][q]
        assert start_id[0] in row["prompt_token_ids"]
        ids, logps = row["token_ids"], np.asarray(row["logprobs"], dtype=np.float64)
        assert len(ids) == len(logps) and np.isfinite(logps).all()
        assert (logps <= 1e-5).all()
        closed = end[0] in ids
        stop_think = ids.index(end[0]) if closed else len(ids)
        first, previous, selected = 0, None, []
        for stop in range(stop_think + 1):
            delimiter = stop < stop_think and ids[stop] in boundary
            if not delimiter and not (stop == stop_think and closed):
                continue
            if stop > first:
                c = float(np.exp(logps[first:stop]).mean())
                v = 0. if previous is None else (c - previous)**2 / 4
                records.append(dict(question=q, start=first, stop=stop,
                    confidence=c, variance=v, lexical_hit=bool(
                        author.has_lexicon_hit(tok.decode(ids[first:stop])))))
                selected.append(first + len(row["prompt_token_ids"]))
                previous = c
            first = stop + 1
        positions.append(dict(question=q, positions=selected, think_stop=stop_think))
    c = np.asarray([r["confidence"] for r in records])
    v = np.asarray([r["variance"] for r in records])
    cl, ch = np.quantile(c, [.25, .75])
    vl, vh = np.quantile(v, [.25, .75])
    assert 0 <= cl < ch < 1 and 0 <= vl < vh <= .25
    save(out / "steps.json", records)
    save(out / "positions.json", positions)
    save(out / "protocol.json", dict(version="auto-code-v2", model=str(args.model),
        source=str(source), source_sha256=sha(source / "generations.jsonl"),
        questions=500, steps=len(records), confidence_quantiles=[cl, ch],
        variance_quantiles=[vl, vh], seed=42,
        confidence="arithmetic mean of raw max probabilities, offline and online",
        steps_policy="original token boundaries; no delimiter or incomplete trailing step",
        classification="author lexical OR low confidence; nonlexical AND high confidence",
        layer_selection="PCA64 + Ridge alpha1; question-group 80/20 split; seed42",
        split_check=split))
    print(f"Prepared {len(records)} steps, c=({cl}, {ch}), v=({vl}, {vh})", flush=True)


def collect(args):
    from transformers import AutoModelForCausalLM
    out = args.output
    features = (args.feature_dir or out).resolve()
    features.mkdir(parents=True, exist_ok=True)
    if (out / "collection.json").exists() or any(features.glob("layer_*.npy")):
        raise FileExistsError("Hidden collection already started")
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["source_sha256"] == sha(args.source / "generations.jsonl")
    positions = json.loads((out / "positions.json").read_text())
    rows = read(args.source / "generations.jsonl")
    count = sum(len(x["positions"]) for x in positions)
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval()
    decoder = model.model
    if not hasattr(decoder, "layers") or not hasattr(decoder, "norm"):
        raise ValueError("Model needs an explicit decoder-output hook adapter")
    layers, width = len(decoder.layers), model.config.hidden_size
    required_bytes = layers * (count * width * 4 + 256)
    new_bytes = required_bytes if not args.feature_cache else required_bytes // layers
    if shutil.disk_usage(features).free < new_bytes + 1024**3:
        raise OSError(f"Hidden cache needs {new_bytes / 1024**3:.2f} GiB "
                      f"plus 1 GiB reserve at {features}")
    reused = []
    if args.feature_cache:
        cache = args.feature_cache
        cp = json.loads((cache / "protocol.json").read_text())
        assert cp["source_sha256"] == protocol["source_sha256"]
        assert Path(cp["model"]).resolve() == args.model.resolve()
        assert json.loads((cache / "positions.json").read_text()) == positions
        # Legacy cache's final entry is post-norm; only internal block outputs match.
        for i in range(1, layers):
            src = cache / f"layer_{i}.npy"
            x = np.load(src, mmap_mode="r")
            assert x.shape == (count, width) and x.dtype == np.float32
            del x
            os.link(src, features / src.name)
            reused.append(i)
    maps, capture, selected = {}, {}, []
    def hook(i):
        def save_layer(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            capture[i] = h[0, selected].detach().float().cpu().numpy()
        return save_layer
    missing = [i for i in range(1, layers+1) if i not in reused]
    for i in missing:
        maps[i] = np.lib.format.open_memmap(features / f"layer_{i}.npy", mode="w+",
            dtype=np.float32, shape=(count, width))
    handles = [decoder.layers[i-1].register_forward_hook(hook(i)) for i in missing]
    offset, checked = 0, False
    started = time.time()
    for q, (row, pos) in enumerate(zip(rows, positions, strict=True)):
        selected = pos["positions"]
        if not selected:
            continue
        ids = row["prompt_token_ids"] + row["token_ids"][:pos["think_stop"]]
        with torch.inference_mode():
            result = decoder(torch.tensor([ids], device="cuda"), use_cache=False,
                return_dict=True, output_hidden_states=not checked)
            if not checked:
                for i in range(1, layers):
                    ref = result.hidden_states[i][0, selected].float().cpu().numpy()
                    actual = (np.load(features / f"layer_{i}.npy", mmap_mode="r")
                              [offset:offset+len(selected)] if i in reused else capture[i])
                    assert np.array_equal(actual, ref), i
                last = torch.from_numpy(capture[layers]).to("cuda", dtype=torch.bfloat16)
                torch.testing.assert_close(decoder.norm(last), result.hidden_states[-1][0, selected],
                                           rtol=0, atol=0)
                save(out / "layer_hook_check.json", dict(exact=True, layers=layers,
                    positions=len(selected), final_output_verified_after_norm=True))
                checked = True
        for i in missing:
            maps[i][offset:offset+len(selected)] = capture.pop(i)
        offset += len(selected)
        del result
        if (q+1) % 50 == 0:
            print(f"Decoder replay {q+1}/500, {time.time()-started:.1f}s", flush=True)
    assert offset == count and checked
    for h in handles:
        h.remove()
    for x in maps.values():
        x.flush()
    save(out / "collection.json", dict(layers=layers, width=width, steps=count,
        feature_dir=str(features), dtype="float32", cache_bytes=required_bytes,
        layer_ids=list(range(1, layers+1)), seconds=time.time()-started,
        representation="raw decoder block outputs (before final norm)", reused_layers=reused))


def select(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from threadpoolctl import threadpool_limits
    out = args.output
    if (out / "selected_layer.json").exists():
        raise FileExistsError("Layer already selected")
    steps = json.loads((out / "steps.json").read_text())
    config = json.loads((out / "collection.json").read_text())
    features = Path(config.get("feature_dir", out))
    y = np.asarray([r["confidence"] for r in steps], dtype=np.float32)
    groups = np.asarray([r["question"] for r in steps])
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                         random_state=42).split(y, groups=groups))
    assert not set(groups[train]) & set(groups[valid])
    scores, started = [], time.time()
    with threadpool_limits(limits=8):
        for layer in config["layer_ids"]:
            x = np.load(features / f"layer_{layer}.npy", mmap_mode="r")
            pca = PCA(n_components=min(64, len(train), x.shape[1]),
                      svd_solver="randomized", random_state=42)
            xt, xv = pca.fit_transform(x[train]), None
            xv = pca.transform(x[valid])
            model = Ridge(alpha=1.).fit(xt, y[train])
            score = float(r2_score(y[valid], model.predict(xv)))
            assert np.isfinite(score)
            scores.append(dict(layer=layer, r2=score))
            save(out / "layer_scores.json", scores)
            print(f"Layer {layer}: held-out-question R2={score:.6f}", flush=True)
            del x, xt, xv, pca, model
    best = max(scores, key=lambda r: (r["r2"], -r["layer"]))
    save(out / "selected_layer.json", dict(best=best, scores=scores,
        seconds=time.time()-started, training_questions=sorted(set(groups[train].tolist())),
        validation_questions=sorted(set(groups[valid].tolist()))))


def fit(args):
    out = args.output
    if (out / "fit.json").exists() or (out / "auto_vector.pt").exists():
        raise FileExistsError("Calibration fit already exists")
    protocol = json.loads((out / "protocol.json").read_text())
    selected = json.loads((out / "selected_layer.json").read_text())
    layer = selected["best"]["layer"]
    steps = json.loads((out / "steps.json").read_text())
    c = np.asarray([r["confidence"] for r in steps])
    lexical = np.asarray([r["lexical_hit"] for r in steps])
    cl, ch = protocol["confidence_quantiles"]
    vl, vh = protocol["variance_quantiles"]
    over, under = lexical | (c < cl), ~lexical & (c > ch)
    assert over.any() and under.any() and not (over & under).any()
    collection = json.loads((out / "collection.json").read_text())
    features = Path(collection.get("feature_dir", out))
    x = np.load(features / f"layer_{layer}.npy", mmap_mode="r")
    xo, xu = x[over].astype(np.float64), x[under].astype(np.float64)
    mo, mu = xo.mean(0), xu.mean(0)
    direction = mo-mu
    var = xo.var(0)+xu.var(0)
    w = direction/(var+1e-12)
    projection = float(w @ direction)
    assert projection > 0 and np.isfinite(w).all()
    # Correct midpoint for the actual decision score w.h - t.
    threshold = float(.5 * w @ (mo+mu))
    moderate = float((w @ mo - threshold)/projection)
    aggressive = float(np.max((xo @ w-threshold)/projection))
    assert np.isclose(moderate, .5) and aggressive >= moderate > 0
    residual = float(np.max(xo @ w - aggressive*projection - threshold))
    assert residual <= 1e-8 * max(1., abs(threshold))
    ceiling = moderate * (1-ch)/(ch-cl)
    # Retain author tau when well inside its feasible interval; otherwise adapt it.
    tau = min(.01, .5*ceiling)
    runtime = load_module("auto_runtime", ROOT /
        "sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py")
    runtime.validate_curve_targets(cl, ch, -moderate, tau)
    params = dict(q25c=cl, q75c=ch, q25v=vl, q75v=vh, low_val_1=-moderate,
        low_val_2=-aggressive, high_val_2=.1, initial_coef=-1., curve_tau=tau)
    control = runtime.ReBalanceParams(boundary_token_ids=(1,), think_start_token_id=2,
                                     think_end_token_id=3, **params)
    z = runtime._curve_constants(cl, ch, -moderate, torch.device("cpu"), torch.float64, tau)
    mid, k, a, b, _, _ = z
    if not 1e-6 < k < 1e6:
        raise ValueError("Curve solver hit search boundary")
    targets = torch.tensor([-moderate, 0., tau], dtype=torch.float64)
    anchors = runtime._baseline(torch.tensor([cl, ch, 1.], dtype=torch.float64), mid, k, a, b)
    torch.testing.assert_close(anchors, targets, rtol=1e-6, atol=1e-9)
    grid_c, grid_v = torch.meshgrid(torch.linspace(0, 1, 501),
                                   torch.linspace(0, .25, 251), indexing="ij")
    surface = runtime.compute_rebalance_coefficient(grid_c, grid_v, control)
    assert torch.isfinite(surface).all()
    tensor = torch.from_numpy(direction.astype(np.float32))
    torch.save(tensor, out / "auto_vector.pt")
    save(out / "curve_check.json", dict(k=k, anchors=anchors.tolist(),
        targets=targets.tolist(), max_residual=float((anchors-targets).abs().max()),
        F_one_ceiling=ceiling, grid_points=surface.numel(), finite=True,
        separator_crossing_residual=residual))
    save(out / "fit.json", dict(version="auto-code-v2", model=str(args.model),
        hidden_state_index=layer, decoder_output_layer=layer-1, parameters=params,
        vector_sha256=sha(out / "auto_vector.pt"), vector_norm=tensor.norm().item(),
        calibration_source_sha256=protocol["source_sha256"], positives=int(over.sum()),
        negatives=int(under.sum()), selection=selected["best"],
        method="Author mixed labels/raw vector/online arithmetic controller; automatic "
               "layer selection with question-group validation; original-token alignment; "
               "correct LDA midpoint sign; low1=-mean crossing, low2=-max crossing; "
               "tau=min(0.01, half the feasible F(1) ceiling). Fixed high2=0.1, initial=-1. "
               "These are explicit adaptation choices, not an exact author reproduction."))
    print((out / "fit.json").read_text(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["generate", "prepare", "collect", "select", "fit"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--feature-dir", type=Path,
                   help="Optional scratch storage; metadata and fitted assets stay in output")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        args.source.mkdir(parents=True, exist_ok=True)
        generate(args.source, str(args.model))
    else:
        globals()[args.stage](args)
