"""Execute the precommitted CPU triage once. This does not generate answers."""
import argparse
import hashlib
import json
import math
import subprocess
import tarfile
import time

import numpy as np

from mechanism_candidates import (ROOT, BASE, require, sha, read, save, read_vector,
    write_vector, bfloat16, cosine, moments, directions, checkpoint_cost, zero_loss_upper)

PLAN = ROOT/BASE/'configs/mechanism_cpu_plan_20260911.json'


def proxy_diagnostic(steps, rows, validation):
    """Forward-in-time previous-step borrowing; future half is a diagnostic label."""
    by_question = {}
    for s in steps:
        by_question.setdefault(s['question'], []).append(s)
    errors = {'short': [], 'all': []}
    counts = {'short': 0, 'all': 0}
    for q in sorted(validation):
        probs = np.exp(np.asarray(rows[q]['logprobs'], dtype=np.float64))
        per_question = {'short': [], 'all': []}
        previous = None
        for s in by_question[q]:
            values = probs[s['start']:s['stop']]
            require(len(values) and abs(values.mean()-s['confidence']) < 1e-12, 'Step confidence mismatch')
            if len(values) >= 2 and previous is not None:
                k = len(values)//2
                target = values[k:].mean()
                plain = values[:k].mean()
                pool = np.concatenate((previous[-8:], values[:k])).mean()
                loss = ((plain-target)**2, (pool-target)**2)
                per_question['all'].append(loss)
                if len(values) <= 8:
                    per_question['short'].append(loss)
            previous = values
        for name, losses in per_question.items():
            if losses:
                errors[name].append(np.asarray(losses).mean(axis=0))
                counts[name] += len(losses)
    result = {}
    for name, values in errors.items():
        require(bool(values), 'No proxy observations: '+name)
        a, b = np.mean(values, axis=0)
        result[name] = dict(questions=len(values), steps=counts[name], plain_mse=float(a), pooled_mse=float(b),
                            ratio=float(b/a), per_question_improved=sum(b<a for a,b in values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output).resolve()
    require(not output.exists(), 'Do not overwrite or rerun an existing audit')
    plan = read(PLAN)
    require(plan['status'] == 'fixed_before_cpu_statistics', 'Plan is not fixed')
    inputs = plan['inputs']; backup = ROOT/inputs['backup']
    manifest = read(ROOT/inputs['backup_manifest'])
    verified = {}
    for name, expected in manifest['files'].items():
        path = backup/name
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Backup changed: '+name)
        verified[name] = expected
    fit, protocol, selection = [read(backup/name) for name in ('fit.json','protocol.json','selected_layer.json')]
    steps = read(backup/'steps.json')
    protected = {name:sha(ROOT/name, source=True) for name in (
        BASE+'configs/final_results_20260909.json', BASE+'scripts/calibrate_auto.py',
        'sources/ReBalance/hidden_analysis_auto.py', 'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py')}
    rows = []
    with tarfile.open(ROOT/inputs['generations_archive']) as archive:
        with archive.extractfile(inputs['generations_member']) as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == inputs['generations_sha256'], 'Saved answers changed')
        for line in archive.extractfile(inputs['generations_member']):
            rows.append(json.loads(line))
    require(len(rows) == 500 and len(steps) == 84008, 'Unexpected calibration shape')
    q = np.array([s['question'] for s in steps]); c = np.array([s['confidence'] for s in steps])
    lexical = np.array([s['lexical_hit'] for s in steps], dtype=bool)
    over = lexical | (c < fit['parameters']['q25c'])
    under = ~lexical & (c > fit['parameters']['q75c'])
    require(int(over.sum()) == 48251 and int(under.sum()) == 11309, 'Labels changed')
    train, valid = [set(selection[key]) for key in ('training_questions','validation_questions')]
    require(len(train) == 400 and len(valid) == 100 and train|valid == set(range(500)) and not train&valid, 'Wrong groups')
    output.mkdir(parents=True)
    save(output/'plan.json',plan)
    started = time.time()
    report = dict(status='running_cpu_only', plan_sha256=sha(PLAN, source=True), inputs_verified=verified,
        source_sha256={BASE+'scripts/'+name:sha(ROOT/BASE/'scripts'/name, source=True) for name in ('mechanism_candidates.py','audit_mechanism_candidates.py')},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        new_generations=0, model_forwards=0, server_connections=0, protected_before=protected,
        limitations=['All 500 saved traces are development data.', 'Local geometry and probability prediction are not generated compression or accuracy.', 'The original layer-selection fold is reused and is not an independent confirmation.'])
    save(output/'summary.json',report)
    x = np.load(backup/'layer_21.npy', mmap_mode='r')
    require(x.shape == (84008,1536) and x.dtype == np.float32, 'Unexpected features')
    masks = {}
    for name, group in [('full', np.ones(len(q),dtype=bool)),('train',np.isin(q,list(train))),('validation',np.isin(q,list(valid)))]:
        masks[name+'_over'] = group&over; masks[name+'_under'] = group&under
    stats = moments(x,masks)
    fitted = {name:directions(stats[name+'_over'],stats[name+'_under']) for name in ('full','train','validation')}
    full = fitted['full']; d, w, u = [full[name] for name in ('original','w','u')]
    original = read_vector(backup/'auto_vector.pt',backup/'auto_vector.pt')
    require(np.array_equal(d.astype(np.float32),original), 'Original raw direction not reproduced bitwise')
    threshold = float(.5*w@(stats['full_over']['mean']+stats['full_under']['mean']))
    max_score = -math.inf
    for start in range(0,len(x),2048):
        values = x[start:start+2048][over[start:start+2048]].astype(np.float64)@w
        if len(values): max_score = max(max_score,float(values.max()))
    aggressive = (max_score-threshold)/full['score']
    require(abs(aggressive+fit['parameters']['low_val_2']) < 1e-10, 'Original LDA parameter not reproduced')
    report['reproduction'] = dict(raw_float32_bitwise_equal=True, low_val_2=-aggressive,
        norm=float(np.linalg.norm(d)), readout_score=full['score'], raw_readout_cosine=cosine(d,w),
        background_cosine=cosine(d,u), over=int(over.sum()), under=int(under.sum()),
        selected_features='First token of each complete step, not mean span or answer-boundary features')
    report['vectors'] = {}
    for entry in plan['candidates_in_priority_order'][:2]:
        name, gates = entry['id'],entry['cpu_pass']; vector = full[name]
        if vector is None or any(fitted[group][name] is None for group in ('train','validation')):
            report['vectors'][name] = dict(status='rejected_cpu',reason='Nonpositive score denominator')
            continue
        v32 = vector.astype(np.float32)
        metrics = dict(norm=float(np.linalg.norm(vector)), norm_ratio=float(np.linalg.norm(vector)/np.linalg.norm(d)),
            relative_vector_change=float(np.linalg.norm(vector-d)/np.linalg.norm(d)), cosine_to_original=cosine(d,vector),
            full_train_cosine=cosine(vector,fitted['train'][name]), train_validation_cosine=cosine(fitted['train'][name],fitted['validation'][name]),
            float32_score_relative_error=float(abs(w@v32/(w@original)-1)),
            bfloat16_vector_score_relative_error=float(abs(w@bfloat16(v32)/(w@bfloat16(original))-1)),
            background_projection=float(u@vector), score_denominator_ratio=full['denominator_ratio'])
        checks = {}
        for key, limit in gates.items():
            if key.startswith('min_'): checks[key] = metrics[key[4:]] >= limit
            elif key.startswith('max_'): checks[key] = metrics[key[4:]] <= limit
        passed = all(checks.values())
        metrics.update(status='ready_for_fixed_gpu_screen' if passed else 'rejected_cpu',checks=checks,
                       generation_effect='UNTESTED', parameter_change='vector only; every original controller parameter retained')
        directory = output/name; directory.mkdir()
        np.save(directory/'direction.npy',v32,allow_pickle=False)
        write_vector(directory/'auto_vector.pt',v32,backup/'auto_vector.pt')
        variant_fit = dict(fit, vector_sha256=sha(directory/'auto_vector.pt'), vector_norm=float(np.linalg.norm(v32)),
            variant=name, parent_fit_sha256=sha(backup/'fit.json'), cpu_plan_sha256=report['plan_sha256'],
            method=entry['mechanism']+' Original auto-code-v2 controller parameters retained. Experimental, no generation benefit established.')
        save(directory/'fit.json',variant_fit)
        metrics['assets']={file:dict(sha256=sha(directory/file),bytes=(directory/file).stat().st_size) for file in ('direction.npy','auto_vector.pt','fit.json')}
        report['vectors'][name]=metrics
        print(json.dumps({name:metrics},ensure_ascii=False),flush=True)
    report['short_step_pooling']=proxy_diagnostic(steps,rows,valid)
    short = report['short_step_pooling']; gate = plan['candidates_in_priority_order'][2]['cpu_pass']
    short['checks'] = dict(enough_questions=short['short']['questions']>=gate['min_questions_with_short_steps'],
        short_improves=short['short']['ratio']<=gate['max_short_question_mean_mse_ratio'],
        no_broad_lag=short['all']['ratio']<=gate['max_all_question_mean_mse_ratio'])
    short['status']='proxy_pass_requires_further_preparation' if all(short['checks'].values()) else 'rejected_cpu_proxy'
    costs=[checkpoint_cost(len(row['token_ids'])) for row in rows]
    # Schema audit only: original records contain one final answer, not answers
    # forced at every checkpoint. No final-answer correctness is fitted here.
    report['answer_checked_exit']=dict(status='deferred_missing_counterfactuals_and_kv_fork',
        schema_keys=sorted(set().union(*(set(row) for row in rows))),
        existing_checkpoint_forced_answers=False, selected_states_are_answer_boundaries=False,
        checkpoint_policy=dict(start=1024,stride=1024,probe_cap=48),
        average_max_probe_tokens=float(np.mean([r['max_extra_probe_tokens'] for r in costs])),
        average_repeated_prefill_prefix_tokens=float(np.mean([r['repeated_prefill_prefix_tokens'] for r in costs])),
        cost_note='Accounting on old unsteered calibration lengths only; assumes probes never stop. Prefix replay and probe decode are different work and cannot be added as equivalent elapsed time. KV-fork speedup is not established.',
        zero_lost_correct_95percent_upper={str(n):zero_loss_upper(n) for n in (100,200,299,500)},
        bound_note='One fixed policy with iid questions and zero observed lost-correct events. No multiplicity adjustment; adaptive selection or non-iid samples invalidate this simple interpretation. No such new observations exist yet.')
    report['protected_after']={name:sha(ROOT/name,source=True) for name in protected}
    require(report['protected_after']==protected,'Protected assets changed')
    report.update(status='completed_cpu_diagnostic', cpu_seconds=time.time()-started)
    save(output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('status','reproduction','short_step_pooling','answer_checked_exit','cpu_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
