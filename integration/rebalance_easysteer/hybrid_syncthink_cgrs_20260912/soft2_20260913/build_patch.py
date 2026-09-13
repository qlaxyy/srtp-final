"""Produce a hash-bound integration patch; never edit shared runtime files."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = 'c688b9c'
PREFIX = 'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/'


def replace(text, old, new):
    assert text.count(old) == 1, old[:150]
    return text.replace(old, new)


def variants():
    names = [PREFIX + n for n in (
        'sample/hybrid_termination.py', 'sample/sampler.py', 'model_runner.py')]
    names += ['integration/rebalance_easysteer/'
              'hybrid_syncthink_cgrs_20260912/run_batch.py']
    names += ['integration/rebalance_easysteer/'
              'hybrid_syncthink_cgrs_20260912/grade_screen.py']
    old = {n: subprocess.check_output(['git', 'show', BASE + ':' + n],
                                    cwd=ROOT).decode() for n in names}
    new = dict(old)
    n = names[0]
    t = old[n]
    t = replace(t, 'import torch', 'import math\n\nimport torch')
    t = replace(t, '    state = getattr(runner, "hybrid_termination", None)', '''    if config.get("mode") == "soft" and (
            params.seed is None or params.temperature <= 0):
        raise ValueError("S64-soft2 requires a seed and positive temperature")
    state = getattr(runner, "hybrid_termination", None)''')
    t = replace(t, 'if config["mode"] not in ("shadow", "enforce"):',
                'if config["mode"] not in ("shadow", "enforce", "soft"):')
    t = replace(t, '        self.events = []', '''        self.pending_soft = None
        self.first_bias = torch.full_like(self.count, -1)
        self.bias_count = torch.zeros_like(self.count)
        self.filtered_trigger_count = torch.zeros_like(self.count)
        self.events = []''')
    t = replace(t, '        self.first_entropy[slot] = 0', '''        self.first_entropy[slot] = 0
        self.first_bias[slot] = -1
        self.bias_count[slot] = self.filtered_trigger_count[slot] = 0''')
    t = replace(t, '            prompt_length=prompt_len, mode=mode)', '''            prompt_length=prompt_len, mode=mode)
        if mode == "soft":
            self.completed[req_id].update(
                first_bias=int(self.first_bias[slot]),
                bias_count=int(self.bias_count[slot]),
                filtered_trigger_count=int(self.filtered_trigger_count[slot]))''')
    t = replace(t, '    def read_apply(self, logits, batch):', '''    def has_soft(self):
        return any(data[2] == "soft" for data in self.requests.values())

    def apply_after_filter(self, logits):
        """Bias retained end odds after temperature/top-p, preserving support."""
        ticket = self.pending_soft
        self.pending_soft = None
        if ticket is None:
            return logits
        pos, idx, trigger = ticket
        start = end = None
        if logits.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        column = logits[pos, 151649]
        enabled = trigger & torch.isfinite(column)
        self.filtered_trigger_count[idx] += (trigger & ~enabled).long()
        first = enabled & (self.first_bias[idx] < 0)
        self.first_bias[idx] = torch.where(
            first, self.count[idx], self.first_bias[idx])
        self.bias_count[idx] += enabled.long()
        logits[pos, 151649] = torch.where(
            enabled, column.float() + math.log(2.0), column.float()
        ).to(logits.dtype)
        if end is not None:
            end.record()
            self.events.append((start, end))
        return logits

    def read_apply(self, logits, batch):
        self.pending_soft = None''')
    t = replace(t, '        force = trigger & enforce', '''        force = trigger & enforce
        if self.has_soft():
            soft = torch.tensor(
                [self.requests[batch.req_ids[p]][2] == "soft" for p in positions],
                device=f.device)
            self.pending_soft = pos, idx, trigger & soft''')
    # The soft path must not allocate or write a vocabulary-wide hard mask.
    t = replace(t, '''        replacement = torch.full((f.shape[1],), -torch.inf,
                                 device=logits.device, dtype=logits.dtype)
        replacement[151649] = 0
        # Shadow returns the original logits without any copy/rounding.
        if any(self.requests[batch.req_ids[p]][2] == "enforce" for p in positions):
            logits[pos] = torch.where(force[:, None], replacement, logits[pos])''', '''        # Legacy hard action unchanged; soft writes only the end column later.
        if any(self.requests[batch.req_ids[p]][2] == "enforce" for p in positions):
            replacement = torch.full((f.shape[1],), -torch.inf,
                                     device=logits.device, dtype=logits.dtype)
            replacement[151649] = 0
            logits[pos] = torch.where(force[:, None], replacement, logits[pos])''')
    new[n] = t
    n = names[1]
    t = replace(old[n], '''        input_batch: InputBatch,
    ) -> SamplerOutput:''', '''        input_batch: InputBatch,
        *,
        after_filter=None,
    ) -> SamplerOutput:''')
    t = replace(t, '            return_logprobs=return_logprobs,',
                '            return_logprobs=return_logprobs,\n'
                '            after_filter=after_filter,')
    t = replace(t, '''        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:''', '''        return_logprobs: bool = False,
        after_filter=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:''')
    t = replace(t, '        # Sample the next token.', '''        if after_filter is not None:
            if use_flashinfer or self.sampling_states.any_greedy(idx_mapping_np):
                raise ValueError("S64-soft2 requires seeded non-greedy sampling")

        # Sample the next token.''')
    t = replace(t, '''            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(''', '''            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            if after_filter is not None:
                processed_logits = after_filter(processed_logits)
            sampled = gumbel_sample(''')
    new[n] = t
    n = names[2]
    new[n] = replace(old[n], '            sampler_output = self.sampler(logits, input_batch)', '''            if hybrid_ticket is not None and hybrid.has_soft():
                sampler_output = self.sampler(
                    logits, input_batch, after_filter=hybrid.apply_after_filter)
            else:
                sampler_output = self.sampler(logits, input_batch)''')
    n = names[3]
    t = replace(old[n], '''        rows = dataset(role, bool(r.get('expansion')))''', '''        rows = (r['soft2_rows'][role] if r.get('soft2_rows')
                else dataset(role, bool(r.get('expansion'))))''')
    t = replace(t, "            if mode in ('shadow','enforce'):",
                "            if mode in ('shadow','enforce','soft'):")
    t = replace(t, "                    h['worker_end_position']=h['end_position']", '''                    if mode == 'soft':
                        h['worker_first_bias'] = h['first_bias']
                        if h['first_bias'] >= record['tokens']:
                            h['first_bias'] = -1
                    h['worker_end_position']=h['end_position']''')
    t = replace(t, "        if r.get('engineering_reuse'):\n            groups = groups[2:]", '''        if r.get('soft2_rows'):
            groups = [('off_R',True,'off'),('shadow_R',True,'shadow'),
                      ('S',False,'soft'),('RS',True,'soft')]
        if r.get('engineering_reuse'):
            groups = groups[2:]''')
    t = replace(t, "            trigger=y['hybrid']['first_trigger']", '''            trigger=y['hybrid']['first_bias' if r.get('soft2_rows')
                                else 'first_trigger']''')
    t = replace(t, "                group(stage,name,use_r,mode)\n    llm.llm_engine.engine_core.shutdown()", '''                if r.get('soft2_rows') and mode == 'enforce':
                    mode = 'soft'
                group(stage,name,use_r,mode)
    llm.llm_engine.engine_core.shutdown()''')
    new[n] = t
    n = names[4]
    t = replace(old[n], "'math_test','gsm8k_test'])",
                "'math_test','gsm8k_test','math_soft2','gsm8k_soft2'])")
    t = replace(t, "    rows=[json.loads(s) for s in (Path(__file__).parent/data_dir/(a.role+'.jsonl')).read_text(encoding='utf-8').splitlines()]", '''    if a.role.endswith('_soft2'):
        resolved = json.loads((a.output/'resolved_plan.json').read_text(encoding='utf-8'))
        rows = resolved['soft2_rows'][a.role]
    else:
        rows=[json.loads(s) for s in (Path(__file__).parent/data_dir/(a.role+'.jsonl')).read_text(encoding='utf-8').splitlines()]''')
    new[n] = t
    return old, new


def main():
    old, new = variants()
    patch = ''.join(''.join(difflib.unified_diff(
        old[n].splitlines(True), new[n].splitlines(True),
        fromfile='a/' + n, tofile='b/' + n)) for n in new)
    for n, text in new.items():
        ast.parse(text, filename=n, feature_version=(3, 10))
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    manifest = dict(base_commit=subprocess.check_output(
        ['git', 'rev-parse', BASE], cwd=ROOT, text=True).strip(),
        files={n: dict(base_lf_sha256=digest(old[n]),
                       patched_lf_sha256=digest(new[n])) for n in new},
        patch_sha256=digest(patch), default_mode='off', applied=False)
    for path, text in [(HERE/'implementation.patch', patch),
                       (HERE/'patch_manifest.json', json.dumps(
                           manifest, ensure_ascii=False, indent=2) + '\n')]:
        with path.open('w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
