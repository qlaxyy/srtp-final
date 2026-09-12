"""Fix a replay of existing stochastic answers; no answer text enters Git."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from mechanism_candidates import ROOT, BASE, read, save, sha, require


def prompt_body(source):
    module = ast.parse(source)
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt')
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); out = a.output.resolve(); require(not out.exists(), 'Immutable bundle exists')
    hypothesis = ROOT/BASE/'configs/sampled_confidence_20260912.json'
    previous_path = ROOT/BASE/'configs/seal_comparison130_20260912/plan.json'
    previous = read(previous_path)
    relative = '.codex_work/overnight_research_20260912/seal_comparison_all_20260912/seal_comparison130_20260912/math_train_unsteered.json'
    raw_path = ROOT/relative; raw = read(raw_path); ledger = read(raw_path.parent/'ledger.json')
    require(sha(raw_path) == ledger['files_sha256']['math_train_unsteered'], 'Saved unsteered answers changed')
    require(raw['status'] == 'completed' and raw['arm'] == 'unsteered' and raw['dataset'] == 'math_train', 'Wrong saved group')
    require(raw['plan_sha256'] == sha(previous_path, source=True), 'Original plan changed')
    require(len(raw['records']) == len(previous['math_train_indices']) == 100, 'Wrong support')
    train_path = ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    questions = []
    for i, (record, train_index) in enumerate(zip(raw['records'], previous['math_train_indices'], strict=True)):
        ids = record['token_ids']; n = len(ids); end = ids.index(151649) if 151649 in ids else n
        require(record['problem'] == train[train_index]['problem'] and record['gold'] == train[train_index]['answer'], 'Question binding changed')
        require(record['dataset_index'] == i and n == record['tokens'] <= 16000 and end == record['thinking_tokens'], 'Malformed saved record')
        require((record['finish_reason'] == 'length') == (n == 16000), 'Cap semantics changed')
        questions.append(dict(index=i, train_index=train_index, thinking_tokens=end, total_tokens=n,
            finish_reason=record['finish_reason'], token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
    builder = BASE+'eval/rebalance_static_eval.py'
    old = subprocess.check_output(['git', 'show', raw['commit']+':'+builder], cwd=ROOT).decode()
    require(prompt_body(old) == prompt_body((ROOT/builder).read_text(encoding='utf-8')), 'Historical prompt builder changed')
    model = previous['model']; hypotheses = read(hypothesis)
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n', encoding='utf-8', newline='\n')
    save(out/'questions.json', questions)
    sources = [BASE+'scripts/'+name for name in ['prepare_sampled_probability_replay.py',
        'replay_sampled_probabilities.py', 'audit_sampled_probabilities.py', 'mechanism_candidates.py', 'audit_control_alignment.py']]
    sources += [builder, 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py']
    plan = dict(status='prepared_saved100_probability_replay_no_generation', hypothesis_sha256=sha(hypothesis, source=True),
        questions=100, capped_answers_included=8, saved_thinking_tokens=sum(q['thinking_tokens'] for q in questions),
        source_answer_commit=raw['commit'], source_answer_sha256=sha(raw_path), source_plan_sha256=raw['plan_sha256'],
        local_source_answer=relative, remote_source_answer=previous['comparison_output']+'/math_train_unsteered.json',
        question_mapping_sha256=sha(out/'questions.json'), train_sha256=sha(train_path, source=True),
        model=model, model_files_sha256=previous['model_files_sha256'],
        dynamic_parameters=previous['dynamic_parameters'],
        prompt_builder_source=builder, prompt_builder_AST_sha256=hashlib.sha256(prompt_body(old).encode()).hexdigest(),
        runtime=hypotheses['replay_runtime'], CPU_opportunity_gate=hypotheses['CPU_opportunity_gate'],
        output='/root/autodl-tmp/results/easysteer/sampled_probability_replay100_20260912',
        source_sha256={p: sha(ROOT/p, source=True) for p in sources},
        source_tokenization='Exact original prompt builder and frozen tokenizer; CPU prepare stores prompt IDs and roundtrip hashes before CUDA load.',
        expected_probability_array_bytes=sum(q['thinking_tokens'] for q in questions)*12,
        new_answers=0, new_grading=0, limitations=[hypotheses['numerical_limits'],
            'Previously evaluated comparison questions are development diagnostics only; no candidate efficacy evidence from this replay.'])
    require(sum(q['finish_reason'] == 'length' for q in questions) == 8, 'Wrong cap count')
    save(out/'plan.json', plan)
    print({k: plan[k] for k in ['questions', 'saved_thinking_tokens', 'expected_probability_array_bytes', 'new_answers']})


if __name__ == '__main__': main()
