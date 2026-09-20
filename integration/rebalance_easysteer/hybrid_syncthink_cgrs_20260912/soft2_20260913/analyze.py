"""Fixed three-arm paired screening analysis; preserve incomplete arms as failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze_expanded import cp_upper


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def analyze(folder, role):
    plan = read(HERE/'plan.json'); count = plan['counts'][role]
    arms = {}; data = {}; records = {}; hashes = {}
    for arm in ('R','S','RS'):
        p = folder/role/arm/'result.json'; g = p.with_name('author_grade.json')
        raw = p.read_bytes(); value = json.loads(raw); grade = read(g)
        assert grade['input_sha256'] == hashlib.sha256(raw).hexdigest()
        assert value['status'] == 'complete' and value['cap'] == 16000
        rows = value['records']; labels = grade['records']
        assert len(rows) == len(labels) == count
        assert [x['train_index'] for x in rows] == [x['train_index'] for x in labels]
        data[arm] = {k:np.array([x[k] for x in rows]) for k in ('tokens','thinking_tokens','answer_tokens')}
        data[arm]['correct'] = np.array([x['author_correct'] for x in labels],dtype=int)
        records[arm] = rows
        arms[arm] = dict(correct=int(data[arm]['correct'].sum()),
            accuracy=float(data[arm]['correct'].mean()),
            **{'mean_'+k:float(data[arm][k].mean()) for k in ('tokens','thinking_tokens','answer_tokens')},
            capped=int((data[arm]['tokens'] == 16000).sum()),
            generation_seconds=value['generation_seconds'],
            checkpoint_io_seconds=value['checkpoint_io_seconds'],
            startup_seconds=value['startup_seconds'], grading_seconds=grade['seconds'],
            statistics_and_bias_gpu_seconds=value['control']['statistics_and_mask_gpu_ms']/1000,
            extra_model_forwards=value['control']['extra_model_forwards'],
            probe_tokens=value['control']['probe_tokens'],
            worker_bias_count=sum((x['hybrid'] or {}).get('bias_count',0) for x in rows),
            worker_filtered_trigger_count=sum((x['hybrid'] or {}).get('filtered_trigger_count',0) for x in rows),
            worker_trigger_count=sum((x['hybrid'] or {}).get('trigger_count',0) for x in rows),
            accepted_first_bias_requests=sum((x['hybrid'] or {}).get('first_bias',-1)>=0 for x in rows),
            discarded_async_tail_tokens=sum((x['hybrid'] or {}).get('discarded_async_tail_tokens',0) for x in rows))
        hashes[arm] = dict(result=hashlib.sha256(raw).hexdigest(),
                           grade=hashlib.sha256(g.read_bytes()).hexdigest())
    for arm in ('S','RS'):
        assert [(x['train_index'],x['problem'],x['prompt_token_ids']) for x in records[arm]] == [
            (x['train_index'],x['problem'],x['prompt_token_ids']) for x in records['R']]
    sample = np.random.default_rng(20260913).integers(0,count,(10000,count))
    ci = lambda x:np.quantile(x,[.025,.975]).tolist()
    comparisons = {}
    for ref in ('R','S'):
        x = data[ref]; y = data['RS']; d = y['correct']-x['correct']
        harmed = int((d == -1).sum()); gained = int((d == 1).sum())
        # Union bound: exact marginal intervals yield >=95% coverage for net loss.
        harm_upper = cp_upper(harmed,count,alpha=.025)
        gain_lower = 1-cp_upper(count-gained,count,alpha=.025)
        comparisons['RS_minus_'+ref] = dict(
            accuracy_delta_pp=float(100*d.mean()),
            accuracy_delta_pp_ci95=ci(100*d[sample].mean(1)),
            net_loss_probability_upper95_conservative=harm_upper-gain_lower,
            accuracy_2pp_noninferiority_supported=harm_upper-gain_lower<=.02,
            wrong_to_right=[r['train_index'] for r,v in zip(records['R'],d) if v==1],
            right_to_wrong=[r['train_index'] for r,v in zip(records['R'],d) if v==-1],
            token_metrics={k:dict(delta=float((y[k]-x[k]).mean()),
                delta_ci95=ci((y[k]-x[k])[sample].mean(1)),
                percent_change=float(100*(y[k].mean()/x[k].mean()-1)),
                percent_change_ci95=ci(100*(y[k][sample].mean(1)/x[k][sample].mean(1)-1)))
                for k in ('tokens','thinking_tokens')})
    s, r, c = arms['S'], arms['R'], arms['RS']
    dominated = (s['accuracy'] >= c['accuracy'] and s['mean_tokens'] <= c['mean_tokens']
                 and s['mean_thinking_tokens'] <= c['mean_thinking_tokens']
                 and (s['accuracy'] > c['accuracy'] or s['mean_tokens'] < c['mean_tokens']
                      or s['mean_thinking_tokens'] < c['mean_thinking_tokens']))
    passed = (c['accuracy']-r['accuracy'] >= -.02-1e-12
              and c['mean_tokens'] < r['mean_tokens']
              and c['mean_thinking_tokens'] < r['mean_thinking_tokens']
              and c['capped'] <= r['capped'] and c['accepted_first_bias_requests'] > 0
              and not dominated)
    return dict(role=role,n=count,arms=arms,comparisons=comparisons,
                point_screen_pass=passed, dominated_by_S=dominated,
                source_sha256=hashes,
                confirmation_completed=False, synergy_evaluated=False,
                limitations=['One generation seed and small sample',
                             'Bootstrap can degenerate when no outcomes differ; use exact bound too',
                             'GPU statistics timing excludes Python bookkeeping but generation includes it',
                             'Worker bias/trigger counts retain possible discarded async tail'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert read(args.run/'batch_status.json')['status'] == 'complete'
    result = {role:analyze(args.run,role) for role in read(HERE/'plan.json')['roles']}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
