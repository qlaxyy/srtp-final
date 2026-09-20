"""Reproduce a pinned upstream indexing bug without importing its runtime."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from mti_reference import contrast, cue_positions, entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_position')
    env = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.source), 'exec'), env)
    positions = torch.tensor([9, 49, 99])
    official = env['get_position'](positions, 2)[:2]
    expected = cue_positions(positions, [1], [2])
    assert official.tolist() == [10, 11]
    assert expected.tolist() == [50, 51]
    checks = ['upstream_subset_position_mismatch_reproduced']
    assert cue_positions(positions, [2, 0], [1, 3]).tolist() == [100, 10, 11, 12]
    checks.append('selected_order_and_variable_cue_lengths')
    assert cue_positions(positions, [], []).numel() == 0
    checks.append('empty_selection')
    raw = torch.tensor([[1., 2., -1.], [0., 1., 3.], [2., -2., 0.]])
    aux = torch.tensor([[2., 1., -1.]])
    assert contrast(raw, None, [1]) is raw
    assert contrast(raw, None, [1], enabled=True, scale=1) is raw
    assert contrast(raw, None, [], enabled=True) is raw
    checks.append('disabled_scale_one_and_empty_exact_identity')
    out = contrast(raw, aux, [1], enabled=True)
    paper = 1.5 * raw[1].log_softmax(-1) - .5 * aux[0].log_softmax(-1)
    torch.testing.assert_close(out[1].softmax(-1), paper.softmax(-1))
    assert torch.equal(out[[0, 2]], raw[[0, 2]])
    assert torch.equal(raw[1], torch.tensor([0., 1., 3.]))
    checks.append('paper_distribution_equivalence_and_input_isolation')
    torch.testing.assert_close(entropy(torch.zeros(1, 5)), torch.tensor([5.]).log())
    checks.append('full_vocab_entropy_natural_log')
    # A penalty applied only on the main branch BEFORE contrast is amplified.
    penalty = torch.tensor([0., .69314718, 0.])
    before = 1.5 * (raw[1] - penalty) - .5 * aux[0]
    after = out[1] - penalty
    torch.testing.assert_close(after - before, .5 * penalty)
    checks.append('lexical_penalty_order_changes_strength')
    report = {
        'status': 'cpu_reference_only_not_gpu_integrated',
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'upstream_commit': 'ac848621b601fef1b61807513dce3f2023bb0e82',
        'checks_passed': checks,
        'reproduction': {'main_last_positions': positions.tolist(), 'selected_rows': [1],
                         'official_cue_positions': official.tolist(),
                         'required_cue_positions': expected.tolist()},
        'kv_finding': 'Static only: skips cue KV writes while advancing seq_lens and reading cache. No GPU equivalence test performed.',
        'gpu_runs': 0, 'efficacy_claim': None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
