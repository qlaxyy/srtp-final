"""CPU-only intake and frozen input preparation; NOT a GPU-ready runner."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def mask_control():
    # An intentionally leaky positive control must change an earlier query.
    q = np.ones((6, 2), dtype=np.float64)
    k = np.zeros_like(q)
    v = np.arange(12, dtype=np.float64).reshape(6, 2)
    changed = v.copy(); changed[4:] += 100
    def attention(value, causal):
        scores = q @ k.T / np.sqrt(2.)
        if causal:
            scores[np.triu_indices(6, 1)] = -np.inf
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights @ value
    require(np.array_equal(attention(v, True)[:4], attention(changed, True)[:4]), 'Causal control failed')
    require(not np.array_equal(attention(v, False)[:4], attention(changed, False)[:4]), 'Leak detector failed')
    return dict(causal_prefix_equal=True, deliberately_unmasked_future_detected=True,
                scope='NumPy reference only, not installed attention or real extractor')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'archive', 'previous', 'parent-plan', 'tokenizer', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Output exists; no overwrite')
    report = a.report.read_bytes()
    with zipfile.ZipFile(a.archive) as z:
        names = z.namelist()
        matching = [n for n in names if n.endswith('.md')]
        require(len(matching) == 1 and z.read(matching[0]) == report, 'Standalone and archived reports differ')
        archive_members = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names}
    parent = read(a.parent_plan)
    require(sha(a.parent_plan) == 'c21ef694cad5f2d27208293a41211cbef068d05383ded94d9872a1907dd1954e', 'Parent changed')
    require(sha(a.previous/'ledger.json') == '7dec84c2607b642f4119d337739db8ad5237df814ba85dd4fede8fc0e328e57a', 'Prior ledger changed')
    ledger = read(a.previous/'ledger.json')
    for name, digest in ledger['assets_sha256'].items():
        require(sha(a.previous/name) == digest, 'Prior asset changed: '+name)
    require(sha(a.tokenizer) == parent['model_files_sha256']['tokenizer.json'], 'Tokenizer mismatch')
    exact = read(a.previous/'exact_inputs.json')
    short, longer = exact['input_ids'], exact['longer_input_ids']
    require(len(short) == 565 and len(longer) == 597 and longer[:565] == short, 'Input scope')
    require(exact['positions'] == [486, 564] and exact['prompt_tokens'] == 290, 'Index scope')
    tokenizer = read(a.tokenizer)
    vocab = {i: token for token, i in tokenizer['model']['vocab'].items()}
    # Exclude every added token, even those incorrectly marked non-special.
    excluded = {t['id'] for t in tokenizer['added_tokens']}
    excluded.update(i for i, token in vocab.items() if any(s in token for s in ('Ċ', 'ĉ', '<', '>')))
    selected = []
    for token in short:
        if token not in excluded and token not in selected:
            selected.append(token)
        if len(selected) == 2:
            break
    require(len(selected) == 2, 'No legal suffix tokens')
    changed = short + [selected[i % 2] for i in range(32)]
    require(changed != longer and changed[:565] == short, 'Suffix unchanged')
    matrix = [('B-S', 'bfloat16', 'S'), ('B-L', 'bfloat16', 'L'), ('B-C', 'bfloat16', 'C'),
              ('F-S', 'float32', 'S'), ('F-S-repeat', 'float32', 'S'),
              ('F-L', 'float32', 'L'), ('F-C', 'float32', 'C')]
    inputs = dict(S=short, L=longer, C=changed)
    require(sum(len(inputs[key]) for _, _, key in matrix) == 4083, 'Call budget')
    anchors = {}
    for label in ['short', 'long']:
        target = np.load(a.previous/('eager_'+label+'.npy'), allow_pickle=False)
        sparse = np.load(a.previous/('eager_'+label+'_all_layers.npy'), allow_pickle=False)
        require(target.shape == (2,1536) and np.isfinite(target).all(), 'Invalid anchor')
        require(np.array_equal(target, sparse[21]), 'Old layer index mismatch')
        anchors[label] = sha(a.previous/('eager_'+label+'.npy'))
    controls = mask_control()
    a.output.mkdir(parents=True)
    shutil.copyfile(a.report, a.output/'decision_specification.md')
    for label in ['short', 'long']:
        shutil.copyfile(a.previous/('eager_'+label+'.npy'), a.output/('anchor_eager_'+label+'.npy'))
    frozen = dict(pair_id=exact['pair_id'], arm=exact['arm'], prompt_tokens=290,
                  absolute_positions=[486,564], output_relative_positions=[196,274], inputs=inputs,
                  suffix_tokens=[dict(id=i, vocabulary_piece=vocab[i]) for i in selected],
                  neighbors={str(i): [dict(position=j, id=short[j], vocabulary_piece=vocab.get(short[j]))
                             for j in range(i-2, min(i+3, len(short)))] for i in [486,564]})
    save(a.output/'inputs.json', frozen)
    receipt = dict(status='input_preparation_complete_instrumentation_pending', gpu_ready=False,
        gpu_authorized=False, line='A/BCC', report_sha256=sha(a.report), archive_sha256=sha(a.archive),
        archive_members_sha256=archive_members, previous_ledger_sha256=sha(a.previous/'ledger.json'),
        previous_assets_verified=len(ledger['assets_sha256']), anchors_sha256=anchors,
        tokenizer_sha256=sha(a.tokenizer), inputs_sha256=sha(a.output/'inputs.json'),
        baseline_commit=subprocess.check_output(['git','rev-parse','bd24112'], text=True).strip(),
        preparation_base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        script_sha256=sha(__file__), matrix=[dict(run_id=r, dtype=d, input=k) for r,d,k in matrix],
        proposed_input_tokens=4083, new_answers=0, proposed_timeout_seconds=600,
        tolerances=dict(max_absolute=.125, per_position_relative_l2=.002), cpu_mask_control=controls,
        pending=['Installed HF dependency source snapshot and identity audit',
                 'Read-only first-block instrumentation and real extractor isolation tests',
                 'Same-loaded-weight BF16 to FP32 identity checks and local reference capture',
                 'Runner, analyzer, partial-output handling and source-bound execution manifest',
                 'Explicit new GPU batch authorization and shared-server idle check'],
        limits='No SSH, full model, actual torch hook, fitting, generation or online code changes in this preparation.')
    save(a.output/'receipt.json', receipt)
    print(json.dumps(dict(status=receipt['status'], suffix_tokens=frozen['suffix_tokens'],
                         previous_assets_verified=receipt['previous_assets_verified'], gpu_ready=False)))


if __name__ == '__main__':
    main()
