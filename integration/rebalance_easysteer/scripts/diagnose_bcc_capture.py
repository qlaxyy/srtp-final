"""Opt-in diagnosis of one failed prefix; never resumes BCC or changes its gate."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from prepare_bcc import read, save, sha, require


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
            def hook(module, inputs, value):
                h = value[0] if isinstance(value, tuple) else value
                capture['h'] = h[0, prefix['positions']].detach().float().cpu().numpy()
            handle = model.model.layers[20].register_forward_hook(hook)
            values = []
            for label, ids in [('short', prefix['input_ids']), ('repeat', prefix['input_ids']), ('long', prefix['longer_input_ids'])]:
                require(time.perf_counter()-started < 600, 'Deadline')
                with torch.inference_mode():
                    result = model.model(torch.tensor([ids], device='cuda'), use_cache=False,
                                         return_dict=True, output_hidden_states=False)
                del result
                h = capture.pop('h')
                # Persist each capture before judging any numerical discrepancy.
                np.save(a.output/(attention+'_'+label+'.npy'), h)
                require(h.shape == (2,1536) and np.isfinite(h).all(), 'Invalid state')
                values.append(h.astype(np.float64)); ledger['input_tokens'] += len(ids)
            first, repeat, longer = values
            norm = float(np.linalg.norm(first)); require(norm > 1e-12, 'Zero state')
            maximum = float(np.max(np.abs(first-longer)))
            relative = float(np.linalg.norm(first-longer)/norm)
            ledger['checks'].append(dict(attention=attention, repeat_exact=bool(np.array_equal(first, repeat)),
                repeat_max_absolute=float(np.max(np.abs(first-repeat))),
                causal_max_absolute=maximum, causal_relative_l2=relative,
                original_tolerance_pass=maximum <= .125 and relative <= .002))
            save(a.output/'ledger.json', ledger)
            handle.remove(); del model; gc.collect(); torch.cuda.empty_cache()
        require(ledger['input_tokens'] == 3454 and time.perf_counter()-started <= 600, 'Scope/deadline')
        ledger['status'] = 'diagnostic_completed_not_BCC_replay'
    except BaseException as exc:
        ledger.update(status='incomplete', error=repr(exc))
        raise
    finally:
        ledger.update(total_seconds=time.perf_counter()-started,
                      assets_sha256={f.name:sha(f) for f in a.output.glob('*.npy')})
        save(a.output/'ledger.json', ledger)


if __name__ == '__main__':
    main()
