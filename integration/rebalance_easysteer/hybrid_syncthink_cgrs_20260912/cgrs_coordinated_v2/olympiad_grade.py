"""Pinned ReBalance-public-code scoring for Olympiad, including pre-GPU CPU acceptance."""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from engineering import ROOT, HERE, read, save, sha


def libraries():
    sys.path.insert(0, str(ROOT/'sources/ReBalance'))
    from utils.parser import extract_answer, parse_ground_truth
    from utils.grader import check_is_correct
    return extract_answer, parse_ground_truth, check_is_correct


def acceptance(plan_path, output):
    plan = read(plan_path)
    for name, value in plan['source_sha256'].items():
        assert sha(ROOT/name, True) == value, name
    extract, ground, check = libraries()
    bad = []
    for row in plan['rows']:
        _, gold = ground(row, 'olympiad')
        pred = extract('\\boxed{' + gold + '}')
        if not check(pred, gold):
            bad.append(row['source_id'])
    fixtures = [('\\boxed{2}', '2', True), ('\\boxed{3}', '2', False),
                ('\\boxed{\\frac{1}{2}}', '0.5', True), ('\\boxed{-2}', '2', False),
                ('\\boxed{(1,2)}', '(1,2)', True), ('\\boxed{(1,2)}', '(2,1)', False),
                ('\\boxed{[0,1]}', '[0,1]', True), ('No final answer', '2', False)]
    outcomes = [dict(prediction=p, gold=g, expected=e, actual=bool(check(extract(p), g))) for p,g,e in fixtures]
    passed = not bad and all(x['expected']==x['actual'] for x in outcomes)
    save(output, dict(passed=passed, plan_sha256=sha(plan_path), reference_roundtrips=675,
        failed_reference_ids=bad, fixtures=outcomes,
        semantics='ReBalance public parser/grader; not the metadata-aware official OlympiadBench evaluator',
        limitations='Exact-reference roundtrip does not establish perfect semantic grading. Public pipeline ignores original per-row Olympiad tolerance metadata; report this consistently for all arms.'))
    if not passed:
        raise RuntimeError('Public grader acceptance failed; do not start GPU')


def grade(folder, resume):
    from full_grade import compare
    plan = read(folder/'resolved_plan.json')
    assert read(folder/'batch_status.json')['status']=='complete'
    assert not (folder/'analysis.json').exists()
    for name, value in plan['source_sha256'].items():
        assert sha(ROOT/name, True)==value, name
    extract, ground, check = libraries()
    began=time.perf_counter(); groups={}; reused={}
    for arm in plan['arms']:
        f=folder/arm; result=read(f/'result.json'); records=result['records']
        assert len(records)==675 and result['status']=='complete'
        p=f/'author_partial.jsonl'
        previous=[__import__('json').loads(l) for l in p.read_text().splitlines()] if resume and p.exists() else []
        assert len(previous)<=675
        reused[arm]=dict(count=len(previous),sha256=sha(p) if p.exists() else None)
        labels=[]
        with p.open('a' if resume else 'x',encoding='utf8') as stream:
            for i,(r,row) in enumerate(zip(records,plan['rows'])):
                assert r['problem_sha256']==row['problem_sha256'] and r['dataset_index']==i
                assert r['tokens']==len(r['token_ids'])<=16000
                identity=dict(dataset_index=i,source_id=row['source_id'],problem_sha256=row['problem_sha256'],
                              text_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
                if i<len(previous):
                    label=previous[i];assert all(label[k]==v for k,v in identity.items())
                    assert type(label['correct']) is bool
                else:
                    _,gold=ground(row,'olympiad');answer=extract(r['text'])
                    label=dict(identity,correct=bool(check(answer,gold)),extracted_answer=answer)
                    stream.write(__import__('json').dumps(label,ensure_ascii=False)+'\n');stream.flush()
                labels.append(label)
                if (i+1)%100==0 or i==674:print('GRADED',arm,i+1,flush=True)
        values=[float(x[1]) for x in csv.reader((f/'gpu.csv').read_text().splitlines()) if len(x)==4]
        events=result['events']
        groups[arm]=dict(n=675,correct=sum(x['correct'] for x in labels),accuracy_percent=100*sum(x['correct'] for x in labels)/675,
            mean_total_tokens=sum(r['tokens'] for r in records)/675,mean_thinking_tokens=sum(r['thinking_tokens'] for r in records)/675,
            capped=sum(r['finish_reason']=='length' for r in records),generation_seconds=result['generation_seconds'],
            gpu_utilization=dict(samples=len(values),mean_percent=sum(values)/len(values) if values else None),
            events=dict(eligible=sum(e['eligible'] for e in events.values()),interventions=sum(e['changed'] for e in events.values()),
                        intervened_questions=sum(e['changed']>0 for e in events.values())),
            overhead=dict(probes=0,extra_probe_forwards=0,control_gpu_seconds=None,native_replays=result['replay_counts'],
                scheduler_preemptions=result['scheduler_preemptions'],
                replayed_input_tokens=sum(e.get('replay_prefill_tokens',0) for e in result['replay_events']) if arm=='RCnegative' else None,
                checkpoint_io_seconds=result['checkpoint_io_seconds'],setup_seconds=result['setup_seconds']),
            result_sha256=sha(f/'result.json'),labels=labels)
    comparisons={}
    for base,candidate in [('U','R'),('U','RCnegative'),('R','RCnegative')]:
        refs=read(folder/base/'result.json')['records']; recs=read(folder/candidate/'result.json')['records']
        refs=[dict(r,correct=l['correct']) for r,l in zip(refs,groups[base]['labels'])]
        comparisons[candidate+'_vs_'+base]=compare(refs,recs,groups[candidate]['labels'])
    c=comparisons['RCnegative_vs_R']
    passed=(c['point_loss_within_2pp'] and groups['RCnegative']['capped']<=groups['R']['capped']
            and all(v['percent_change']<0 for v in c['lengths'].values()))
    save(folder/'analysis.json',dict(status='complete_same_batch_three_arm',groups=groups,comparisons=comparisons,
        passes_fixed_point_criteria=passed,grade_invocation_seconds=time.perf_counter()-began,reused_labels=reused,
        pure_generation_seconds=sum(g['generation_seconds'] for g in groups.values()),
        plan_sha256=sha(folder/'resolved_plan.json'),bootstrap=dict(repetitions=10000,seed=20260913),
        limitations=['One seed; paired question intervals omit sampling/batch variability',
            'No lexical-only group; no factorial synergy claim',
            'ReBalance-public grading, not original OlympiadBench tolerance-aware evaluation',
            'GPU controls not separately timed; preemption recomputation included in generation',
            'Engineering gate and failed attempts, if any, have separate costs']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-plan',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.check_plan:acceptance(a.check_plan,a.output)
    else:grade(a.output,a.resume)
