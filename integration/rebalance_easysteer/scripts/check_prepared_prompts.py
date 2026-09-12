"""Freeze prompt IDs without model loading; a local encoder is an audit only.

The local ByteLevel BPE implementation is deliberately limited to the frozen
tokenizer schema. It must match 100 prompts actually encoded by the inference
environment. The native mode then checks every newly prepared prompt with the
unchanged production tokenizer and prompt builder, before any model forward.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from mechanism_candidates import ROOT, BASE, read, save, sha, require

WORK = ROOT / '.codex_work/overnight_research_20260912'
TOKENIZER = ROOT / '.codex_work/label_audit_30_20260910/tokenizer.json'
TOKENIZER_SHA = '88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773'
REFERENCE = WORK / 'probability_prompt_vllm_20260912.json'
REFERENCE_SHA = 'c3f916a0365f81cbb6c0617d5a14b0e787afb81daa1c4777e7db8332f50f950d'


@lru_cache(maxsize=1)
def category_classes():
    groups = {'L': [], 'N': []}
    for code in range(sys.maxunicode + 1):
        kind = unicodedata.category(chr(code))[0]
        if kind in groups:
            groups[kind].append(code)
    def compact(codes):
        ranges = []; start = end = codes[0]
        for code in codes[1:]:
            if code == end + 1:
                end = code
            else:
                ranges.append((start, end)); start = end = code
        ranges.append((start, end))
        return ''.join(re.escape(chr(a)) +
                       ('-' + re.escape(chr(b)) if b != a else '')
                       for a, b in ranges)
    return {name: compact(codes) for name, codes in groups.items()}


class FrozenByteBPE:
    def __init__(self, path):
        require(sha(path) == TOKENIZER_SHA, 'Unsupported tokenizer hash')
        data = read(path); model = data['model']
        require(data['normalizer'] == {'type': 'NFC'}, 'Unsupported normalization')
        pre = data['pre_tokenizer']['pretokenizers']
        require(pre[0]['type'] == 'Split' and pre[0]['behavior'] == 'Isolated'
                and not pre[0]['invert'], 'Unsupported split')
        require(pre[1]['type'] == 'ByteLevel' and not pre[1]['add_prefix_space']
                and not pre[1]['use_regex'], 'Unsupported ByteLevel')
        require(model['type'] == 'BPE' and model['dropout'] is None
                and not model['byte_fallback'] and model['unk_token'] is None,
                'Unsupported BPE settings')
        # Rust regex White_Space differs from Python's four extra C0 controls.
        white = '\\t-\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
        categories = category_classes(); letters = categories['L']; numbers = categories['N']
        pattern = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
                   f'[^\\r\\n{letters}{numbers}]?[{letters}]+|[{numbers}]|'
                   f' ?[^{white}{letters}{numbers}]+[\\r\\n]*|'
                   f'[{white}]*[\\r\\n]+|[{white}]+(?![^{white}])|[{white}]+')
        self.pattern = re.compile(pattern)
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = byte_values[:]; missing = 0
        for byte in range(256):
            if byte not in byte_values:
                byte_values.append(byte); chars.append(256 + missing); missing += 1
        self.byte_encoder = dict(zip(byte_values, map(chr, chars)))
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab = model['vocab']; self.inverse = {v: k for k, v in self.vocab.items()}
        self.ranks = {tuple(pair.split(' ')): i for i, pair in enumerate(model['merges'])}
        added = data['added_tokens']
        require(all(not any(t[k] for k in ('single_word', 'lstrip', 'rstrip', 'normalized'))
                    for t in added), 'Unsupported added-token matching')
        self.added = {t['content']: t['id'] for t in added}
        self.added_inverse = {v: k for k, v in self.added.items()}
        self.special_pattern = re.compile('(' + '|'.join(re.escape(t) for t in
            sorted(self.added, key=len, reverse=True)) + ')')

    @lru_cache(maxsize=16384)
    def piece(self, text):
        symbols = [self.byte_encoder[b] for b in text.encode('utf-8')]
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda pair: self.ranks.get(pair, float('inf')))
            if best not in self.ranks:
                break
            merged = []; i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    merged.append(symbols[i] + symbols[i + 1]); i += 2
                else:
                    merged.append(symbols[i]); i += 1
            symbols = merged
        return tuple(self.vocab[s] for s in symbols)

    def encode(self, text):
        ids = []
        for part in self.special_pattern.split(text):
            if part in self.added:
                ids.append(self.added[part]); continue
            part = unicodedata.normalize('NFC', part)
            pieces = self.pattern.findall(part)
            require(''.join(pieces) == part, 'Pre-tokenizer dropped text')
            for piece in pieces:
                ids.extend(self.piece(piece))
        return ids

    def decode(self, ids):
        output = []; pending = bytearray()
        for token in ids:
            if token in self.added_inverse:
                output.append(pending.decode('utf-8')); pending.clear()
                output.append(self.added_inverse[token])
            else:
                pending.extend(self.byte_decoder[c] for c in self.inverse[token])
        output.append(pending.decode('utf-8'))
        return ''.join(output)


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def local_prepare(bundle, output):
    require(not output.exists(), 'Immutable prompt audit already exists')
    require(sha(REFERENCE) == REFERENCE_SHA, 'Native reference changed')
    tokenizer = FrozenByteBPE(TOKENIZER)
    reference = read(REFERENCE)
    saved = read(WORK / 'seal_comparison_all_20260912/seal_comparison130_20260912/math_train_unsteered.json')
    require(sha(WORK / 'seal_comparison_all_20260912/seal_comparison130_20260912/math_train_unsteered.json') ==
            '238380ad32cbfe84a3dab2cbfd5eed90dcf27889c02014768bc173577d8d1e97', 'Saved prompt source changed')
    require(len(reference['records']) == len(saved['records']) == 100, 'Wrong reference size')
    template = None
    for i, (record, source) in enumerate(zip(reference['records'], saved['records'])):
        require(record['index'] == i, 'Reference order differs')
        prompt = tokenizer.decode(record['prompt_token_ids'])
        require(text_sha(prompt) == record['prompt_text_sha256'], 'Reference decode differs')
        require(prompt.count(source['problem']) == 1, 'Ambiguous problem in native prompt')
        prefix, suffix = prompt.split(source['problem'])
        if template is None:
            template = (prefix, suffix)
        require(template == (prefix, suffix), 'Native prompt template varies')
        encoded = tokenizer.encode(prompt)
        require(encoded == record['prompt_token_ids'], 'Local/native prompt BPE differs at reference ' + str(i))
    plan = read(bundle / 'plan.json'); records = {}; by_index = {}
    for candidate, stages in plan['candidates'].items():
        for stage, entry in stages.items():
            path = bundle / entry['bundle'] / 'questions.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            items = []
            for row in rows:
                require(not any(t in row['problem'] for t in tokenizer.added), 'Added token literal in new problem')
                prompt = template[0] + row['problem'] + template[1]
                ids = tokenizer.encode(prompt)
                require(ids.count(151646) == 1 and ids[0] == 151646, 'BOS changed')
                require(len(ids) + 16000 <= 32768, 'Context budget exceeded')
                item = dict(train_index=row['train_index'], prompt_text_sha256=text_sha(prompt), prompt_token_ids=ids)
                if row['train_index'] in by_index:
                    require(by_index[row['train_index']] == item, 'Shared prompt differs')
                by_index[row['train_index']] = item; items.append(item)
            records[candidate + '/' + stage] = items
    require(len(by_index) == 608, 'Wrong unique prompt count')
    result = dict(status='local_BPE_matches_100_native_prompts_new_inputs_frozen_not_model_validation',
        tokenizer_sha256=TOKENIZER_SHA, native_reference_sha256=REFERENCE_SHA,
        native_reference_count=100, unique_new_prompts=608, template=dict(prefix=template[0], suffix=template[1]),
        unicode_version=unicodedata.unidata_version, prompt_token_count_min=min(len(r['prompt_token_ids']) for r in by_index.values()),
        prompt_token_count_max=max(len(r['prompt_token_ids']) for r in by_index.values()),
        records=records, model_loads=0, GPU_calls=0, new_answers=0,
        limitation='Local audit of this frozen tokenizer only. Native production tokenizer must independently reproduce all new prompt IDs before model loading.')
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'template')}))


def native_check(bundle, output):
    require(not output.exists(), 'Fresh native prompt receipt required')
    parent = read(bundle / 'plan.json'); expected = parent['prompt_audit']
    audit_path = bundle / expected['file']
    require(sha(audit_path) == expected['sha256'], 'Frozen prompt audit changed')
    frozen = read(audit_path)
    replay = read(bundle / parent['probability_replay']['bundle'] / 'plan.json')
    for name, digest in replay['model_files_sha256'].items():
        require(sha(Path(replay['model']) / name) == digest, 'Model asset changed: ' + name)
    from replay_sampled_probabilities import prompt_function
    from vllm.tokenizers.registry import get_tokenizer
    tokenizer = get_tokenizer(replay['model'], local_files_only=True)
    build_prompt = prompt_function(replay); seen = set(); checked = 0
    for candidate, stages in parent['candidates'].items():
        for stage, entry in stages.items():
            rows = [json.loads(line) for line in (bundle / entry['bundle'] / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
            recorded = frozen['records'][candidate + '/' + stage]
            require(len(rows) == len(recorded), 'Prompt record size differs')
            for row, record in zip(rows, recorded, strict=True):
                require(row['train_index'] == record['train_index'], 'Prompt order differs')
                prompt = build_prompt(tokenizer, row['problem']); ids = tokenizer.encode(prompt)
                require(text_sha(prompt) == record['prompt_text_sha256'] and
                        ids == record['prompt_token_ids'], 'Native/local new prompt differs')
                require(len(ids) + 16000 <= 32768, 'Native context overflow')
                seen.add(row['train_index']); checked += 1
    require(len(seen) == 608 and checked == 616, 'Wrong prompt population')
    result = dict(status='all_new_prompts_match_native_production_tokenizer',
        plan_sha256=sha(bundle / 'plan.json'), prompt_audit_sha256=sha(audit_path),
        unique_prompts=len(seen), stage_entries=checked, model_loads=0, CUDA_calls=0)
    save(output, result); print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-check', action='store_true')
    args = parser.parse_args()
    if args.native_check:
        native_check(args.bundle, args.output)
    else:
        local_prepare(args.bundle, args.output)


if __name__ == '__main__':
    main()
