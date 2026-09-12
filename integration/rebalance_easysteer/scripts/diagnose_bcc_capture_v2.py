"""Opt-in diagnosis of one failed prefix; never resumes BCC or changes its gate."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from prepare_bcc import read, save, sha, require


def position_metrics(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    require(first.shape == second.shape and first.ndim == 2, 'State shape')
    require(np.isfinite(first).all() and np.isfinite(second).all(), 'Nonfinite metric input')
    rows = []
    for a, b in zip(first, second):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        require(min(na, nb) > 1e-12, 'Zero state')
        coordinate = int(np.argmax(np.abs(a-b)))
        absolute = float(abs(a[coordinate]-b[coordinate]))
        relative = float(np.linalg.norm(a-b)/na)
        rows.append(dict(max_absolute=absolute, coordinate=coordinate,
                         first_value=float(a[coordinate]), second_value=float(b[coordinate]),
                         first_norm=na, second_norm=nb, relative_l2=relative,
                         cosine=float(a@b/(na*nb)), exact=bool(np.array_equal(a,b)),
                         original_tolerance_pass=absolute <= .125 and relative <= .002))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--expected-plan-sha256')
    a = p.parse_args(); plan = read(a.prepared/'plan.json')
    require(sha(a.prepared/'plan.json') == 'c21ef694cad5f2d27208293a41211cbef068d05383ded94d9872a1907dd1954e', 'Changed parent plan')
    require(sha(a.prepared/'prefixes.json') == plan['prefixes_sha256'], 'Changed prefixes')
    rows = read(a.prepared/'prefixes.json')
    row = next(r for r in rows if r['pair_id'] == 'lexical_direction:4161')
    prefix = next(v for v in row['prefixes'] if v['arm'] == 'lexical_direction')
    require((len(prefix['input_ids']), len(prefix['longer_input_ids'])) == (565, 597), 'Wrong diagnostic scope')
    require(prefix['positions'] == [486, 564], 'Absolute positions changed')
    require(prefix['longer_input_ids'][:565] == prefix['input_ids'], 'Prefix changed')
    require(not a.output.exists(), 'Output exists')
    if not a.execute:
        print('CPU preflight passed: one prefix, two attention implementations, six forwards, 3454 input tokens, no answers')
        return
    require(a.expected_plan_sha256 == sha(a.prepared/'plan.json'), 'Explicit plan hash required')
    started = time.perf_counter()
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy')
    for name, digest in plan['model_files_sha256'].items():
        require(sha(Path(plan['model'])/name) == digest, 'Model changed: '+name)
    a.output.mkdir(parents=True)
    save(a.output/'exact_inputs.json', dict(pair_id=row['pair_id'], prompt_tokens=290, **prefix))
    ledger = dict(status='started', new_answers=0, input_tokens=0,
                  parent_plan_sha256=sha(a.prepared/'plan.json'), script_sha256=sha(Path(__file__)), checks=[])
    save(a.output/'ledger.json', ledger)
    try:
        import gc
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        ledger.update(torch=torch.__version__, transformers=transformers.__version__)
        for attention in ['sdpa', 'eager']:
            require(time.perf_counter()-started < 600, 'Deadline')
            model = AutoModelForCausalLM.from_pretrained(plan['model'], torch_dtype=torch.bfloat16,
                attn_implementation=attention, local_files_only=True).cuda().eval()
            capture = {}
            require(not any(m.training for m in model.modules()), 'Training mode')
            modules = model.model.layers
            ledger.setdefault('attention_modules', {})[attention] = [type(layer.self_attn).__name__ for layer in modules]
            require(all(float(layer.self_attn.attention_dropout) == 0 for layer in modules), 'Nonzero configured dropout')
            def make_hook(index):
                def hook(module, inputs, value):
                    h = value[0] if isinstance(value,tuple) else value
                    capture[index] = h[0,prefix['positions']].detach().float().cpu().numpy()
                return hook
            handles = [layer.register_forward_hook(make_hook(i)) for i,layer in enumerate(modules)]
            handles.append(model.model.embed_tokens.register_forward_hook(make_hook(-1)))
            functional = torch.nn.functional
            original_sdpa = functional.scaled_dot_product_attention
            sdpa_calls = []
            def observed_sdpa(*args, **kwargs):
                dropout = kwargs.get('dropout_p', args[4] if len(args)>4 else 0.)
                require(float(dropout) == 0, 'Actual SDPA dropout nonzero')
                sdpa_calls.append(dict(dropout=float(dropout), is_causal=bool(kwargs.get('is_causal',args[5] if len(args)>5 else False)),
                                       mask_present=(kwargs.get('attn_mask',args[3] if len(args)>3 else None) is not None)))
                return original_sdpa(*args, **kwargs)
            functional.scaled_dot_product_attention = observed_sdpa
            values = []
            for label, ids in [('short', prefix['input_ids']), ('repeat', prefix['input_ids']), ('long', prefix['longer_input_ids'])]:
                require(time.perf_counter()-started < 600, 'Deadline')
                with torch.inference_mode():
                    result = model.model(torch.tensor([ids], device='cuda'), use_cache=False,
                                         return_dict=True, output_hidden_states=False)
                del result
                h = capture[20]
                sparse = np.stack([capture[i] for i in range(-1,len(modules))])
                np.save(a.output/(attention+'_'+label+'_all_layers.npy'), sparse)
                capture.clear()
                # Persist each capture before judging any numerical discrepancy.
                np.save(a.output/(attention+'_'+label+'.npy'), h)
                require(h.shape == (2,1536) and np.isfinite(h).all(), 'Invalid state')
                values.append(sparse.astype(np.float64)); ledger['input_tokens'] += len(ids)
            first, repeat, longer = values
            functional.scaled_dot_product_attention = original_sdpa
            require((len(sdpa_calls) == 3*len(modules)) if attention == 'sdpa' else not sdpa_calls, 'Attention path mismatch')
            layers = [dict(layer=i-1, repeat=position_metrics(first[i],repeat[i]),
                           extension=position_metrics(first[i],longer[i])) for i in range(len(first))]
            ledger['checks'].append(dict(attention=attention, layers=layers,
                sdpa_calls=sdpa_calls, backend_note='Functional SDPA calls observed; automatic CUDA subkernel identity not measured.',
                target_layer20_pass=all(r['original_tolerance_pass'] for r in layers[21]['extension'])))
            save(a.output/'ledger.json', ledger)
            for handle in handles: handle.remove()
            del modules, handles, model; gc.collect(); torch.cuda.empty_cache()
        require(ledger['input_tokens'] == 3454 and time.perf_counter()-started <= 600, 'Scope/deadline')
        ledger['status'] = 'diagnostic_completed_not_BCC_replay'
    except BaseException as exc:
        ledger.update(status='incomplete', error=repr(exc))
        raise
    finally:
        ledger.update(total_seconds=time.perf_counter()-started,
                      assets_sha256={f.name:sha(f) for f in a.output.iterdir() if f.name != 'ledger.json' and f.is_file()})
        save(a.output/'ledger.json', ledger)


if __name__ == '__main__':
    main()
