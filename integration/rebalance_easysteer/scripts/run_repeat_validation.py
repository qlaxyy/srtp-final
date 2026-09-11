"""Inspect a fixed paired plan on CPU. --execute explicitly starts both GPU arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = 'integration/rebalance_easysteer/'
PYTHON = '/root/autodl-tmp/venvs/easysteer-vllm026/bin/python'
GRADER = '/root/autodl-tmp/venvs/rebalance/bin/python'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path, source=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if source else raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def validate_bundle(bundle):
    plan = read(bundle/'plan.json')
    require(plan['status'] == 'prepared_not_run' and plan['count'] == 100, 'Not the fixed 100 plan')
    require(plan['arms'] == {'original_dynamic':'off', 'repeat_cancel_positive':'cancel_positive'}, 'Wrong arms')
    require(plan['runtime']['max_tokens'] == 16000, 'Both arms require cap 16000')
    require(sha(bundle/'validation100.jsonl') == plan['dataset_sha256'], 'Dataset hash mismatch')
    rows = [json.loads(s) for s in (bundle/'validation100.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows) == 100 and [r['train_index'] for r in rows] == plan['train_indices'], 'Question order mismatch')
    require(not any(plan['overlap_checks'].values()), 'Overlapping selection')
    for name, expected in plan['source_sha256'].items():
        require(sha(ROOT/name, source=True) == expected, 'Source changed: '+name)
    return plan, rows


def command(plan, dataset, output, mode):
    assets = Path(plan['assets'])
    cmd = [PYTHON, '-u', BASE+'eval/rebalance_dynamic_eval.py',
           '--model', plan['model'], '--dataset', str(dataset), '--limit', '100',
           '--vector', str(assets/'auto_vector.pt'), '--calibration-fit', str(assets/'fit.json'),
           '--diagnostic-group', 'rebalance_dynamic', '--repeat-gate', mode,
           '--output', str(output)]
    for key, value in plan['runtime'].items():
        flag = '--'+key.replace('_','-')
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def validate_arm(saved, plan, rows, mode):
    require(saved.get('status') == 'diagnostic_completed', 'Incomplete arm; preserve partial output')
    require('baseline' not in saved, 'An unsteered arm was run accidentally')
    protocol = saved['protocol']
    for key, value in plan['runtime'].items():
        require(protocol[key] == value, 'Runtime mismatch: '+key)
    require(protocol['offset'] == 0 and protocol['limit'] == 100, 'Wrong evaluation slice')
    require(protocol['model'] == plan['model'], 'Wrong model')
    require(protocol['easysteer_output_layer'] == plan['decoder_output_layer'], 'Wrong layer')
    require(protocol['run_order'] == ['rebalance_dynamic'], 'Wrong arm type')
    require(protocol.get('repeat_gate', {}).get('mode', 'off') == mode, 'Gate mode mismatch')
    for key, value in plan['dynamic_parameters'].items():
        require(protocol['dynamic_params'][key] == value, 'Controller changed: '+key)
    require(not protocol['dynamic_params'].get('inject_first_step', False), 'Unexpected first-prompt injection')
    for key in ('dataset_sha256', 'vector_sha256'):
        require(saved['provenance'][key] == plan[key], 'Asset mismatch: '+key)
    require(saved['provenance']['calibration_fit_sha256'] == plan['fit_sha256'], 'Fit changed')
    records = saved['rebalance_dynamic']['records']
    require(len(records) == len(rows) == 100, 'Not all 100 answers completed')
    for i, (record, row) in enumerate(zip(records, rows, strict=True)):
        require(record['dataset_index'] == i and record['problem'] == row['problem'], 'Record pairing mismatch')
        require(record['gold'] == row['answer'], 'Gold answer mismatch')
        require(record['tokens'] == len(record['token_ids']) <= 16000, 'Invalid length accounting')
        require(record['finish_reason'] in ('stop', 'length'), 'Unfinished record')
    if mode != 'off':
        require('repeat_gate' in saved['rebalance_dynamic']['summary'], 'Gate metrics missing')
    return saved['rebalance_dynamic']


def combine(original, candidate, plan, rows):
    a = validate_arm(original, plan, rows, 'off')
    b = validate_arm(candidate, plan, rows, 'cancel_positive')
    for key in ('environment', 'calibration'):
        require(original[key] == candidate[key], 'Arms differ in '+key)
    for key in ('commit', 'git_status'):
        require(original['provenance'][key] == candidate['provenance'][key], 'Arms differ in '+key)
    # The legacy grader expects these keys. Their meanings are explicitly recorded.
    protocol = dict(original['protocol'], diagnostic_only=False,
                    run_order=['baseline', 'rebalance_dynamic'],
                    group_meanings={'baseline':'original dynamic; NOT unsteered',
                                    'rebalance_dynamic':'same dynamic plus repeat positive gate'},
                    comparison_scope='fresh training validation, not frozen benchmark',
                    arm_protocols={'original_dynamic':original['protocol'],
                                   'repeat_cancel_positive':candidate['protocol']})
    return dict(status='completed', scope='repeat gate paired training validation',
                protocol=protocol, baseline=a, rebalance_dynamic=b,
                environment=original['environment'],
                provenance=dict(original_dynamic=original['provenance'],
                                repeat_cancel_positive=candidate['provenance']))


def analyze(pair, graded):
    require(set(graded['groups']) == {'baseline','rebalance_dynamic'}, 'Incomplete author grading')
    result = dict(status='completed', scope=pair['scope'], groups={})
    for name, label in (('baseline','original_dynamic'), ('rebalance_dynamic','repeat_cancel_positive')):
        group, grade = pair[name], graded['groups'][name]
        records, summary = group['records'], group['summary']
        require(len(grade['records']) == len(records) == 100, 'Grading count mismatch')
        result['groups'][label] = dict(
            count=100, author_correct=grade['author_correct'],
            author_accuracy_percent=grade['author_correct'],
            mean_total_tokens=sum(r['tokens'] for r in records)/100,
            mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/100,
            capped=sum(r['finish_reason']=='length' or r['tokens']==16000 for r in records),
            generation_seconds=summary['generation_seconds'],
            grading_errors=summary['grading_errors'], preemptions=summary['preemptions'],
            dynamic_kv_replay=summary['dynamic_kv_replay'])
    a, b = result['groups'].values()
    gate = pair['rebalance_dynamic']['summary']['repeat_gate']
    result.update(improved_indices=graded['improved_indices'],
                  degraded_indices=graded['degraded_indices'], gate=gate,
                  total_token_change_percent=100*(b['mean_total_tokens']/a['mean_total_tokens']-1),
                  accuracy_change_percentage_points=b['author_correct']-a['author_correct'],
                  changed_outputs=sum(x['token_ids'] != y['token_ids'] for x,y in zip(pair['baseline']['records'],pair['rebalance_dynamic']['records'],strict=True)))
    if not gate['changed_scales']:
        conclusion = 'No gate action: coverage result only; no effectiveness claim.'
    elif b['mean_total_tokens'] >= a['mean_total_tokens'] or b['author_correct'] < a['author_correct']:
        conclusion = 'Do not expand or retune this validation set.'
    else:
        conclusion = 'Promising on this one 100-question seed only; discuss a separate held-out test. Accuracy preservation is not established.'
    result['decision'] = conclusion
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicit GPU execution; default only inspects local files')
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    plan, rows = validate_bundle(bundle)
    commands = {arm:command(plan, bundle/'validation100.jsonl', output/(arm+'.json'), mode)
                for arm,mode in plan['arms'].items()}
    if not args.execute:
        print(json.dumps(dict(status='cpu_preflight_passed_no_generation', count=100,
                              dataset_sha256=plan['dataset_sha256'], commands=commands), indent=2))
        return
    require(sys.platform == 'linux', 'Use the existing Linux server environment')
    require(not output.exists(), 'New output directory required; no automatic rerun or overwrite')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'Commit the experimental worktree first')
    for executable in (PYTHON, GRADER):
        require(Path(executable).is_file(), 'Existing runtime missing: '+executable)
    # Editable installs may still point at another checkout. Verify the code
    # Python actually imports before loading any model; do not reinstall it.
    import_check = (
        "import inspect,json,vllm,torch; "
        "from easysteer.vectors import from_pt_direction; "
        "print(json.dumps(dict(vllm_file=vllm.__file__, "
        "vector_file=inspect.getfile(from_pt_direction), "
        "vllm_version=vllm.__version__,torch_version=torch.__version__)))"
    )
    import_env = dict(os.environ, PYTHONNOUSERSITE='1')
    receipt = json.loads(subprocess.check_output(
        [PYTHON, '-c', import_check], cwd=ROOT, env=import_env,
        text=True).splitlines()[-1])
    for key, relative in (
        ('vllm_file','sources/EasySteer/vllm-steer/vllm/__init__.py'),
        ('vector_file','sources/EasySteer/easysteer/vectors.py'),
    ):
        require(Path(receipt[key]).resolve() == (ROOT/relative).resolve(),
                'Editable runtime points to a different checkout: '+receipt[key])
    for name, expected in plan['model_files_sha256'].items():
        require(sha(Path(plan['model'])/name) == expected, 'Model changed: '+name)
    for name, expected in (('auto_vector.pt',plan['vector_sha256']), ('fit.json',plan['fit_sha256'])):
        require(sha(Path(plan['assets'])/name) == expected, 'Calibration asset changed: '+name)
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONNOUSERSITE='1', VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env['PATH'] = str(Path(PYTHON).parent)+os.pathsep+env['PATH']
    ledger = dict(status='running', started_unix=time.time(), plan_sha256=sha(bundle/'plan.json'),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  commands=commands, imported_runtime=receipt, arms={})
    save(output/'run_ledger.json', ledger)
    try:
        for arm in plan['run_order']:
            started = time.time()
            with (output/(arm+'.log')).open('x') as log:
                completed = subprocess.run(commands[arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            ledger['arms'][arm] = dict(exit_code=completed.returncode, process_seconds=time.time()-started)
            save(output/'run_ledger.json', ledger)
            require(completed.returncode == 0, 'Arm failed; retained raw/partial outputs in '+str(output))
            validate_arm(read(output/(arm+'.json')), plan, rows, plan['arms'][arm])
        original, candidate = [read(output/(arm+'.json')) for arm in plan['run_order']]
        pair = combine(original, candidate, plan, rows)
        pair['arm_files_sha256'] = {arm:sha(output/(arm+'.json')) for arm in plan['run_order']}
        pair['plan_sha256'] = ledger['plan_sha256']
        save(output/'paired.json', pair)
        grader_cmd = [GRADER, '-u', BASE+'scripts/regrade_saved_results.py',
                      '--input', str(output/'paired.json'), '--output', str(output/'author_grading.json'),
                      '--data-name', 'math', '--group-budget-seconds', '0']
        ledger['grading_command'] = grader_cmd
        save(output/'run_ledger.json', ledger)
        with (output/'author_grading.log').open('x') as log:
            subprocess.run(grader_cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        graded = read(output/'author_grading.json')
        require(graded['input_sha256'] == sha(output/'paired.json'), 'Grading source mismatch')
        require(graded['dataset_sha256'] == plan['dataset_sha256'], 'Grading dataset mismatch')
        analysis = analyze(pair, graded)
        analysis['hashes'] = {name:sha(output/name) for name in ('paired.json','author_grading.json')}
        save(output/'analysis.json', analysis)
        ledger.update(status='completed', completed_unix=time.time())
    except BaseException as error:
        ledger.update(status='incomplete', error=repr(error), stopped_unix=time.time())
        raise
    finally:
        save(output/'run_ledger.json', ledger)
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == '__main__':
    main()
