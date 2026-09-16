"""Fixed deployed-workflow repeatability; no controller or threshold change."""
import hashlib
from engineering import ROOT,HERE,read,sha
ARMS=('R','RC14','RChistory')
SEEDS=(142,242,342)

def schedule():
    return [dict(name=f's{seed}_{arm}',seed=seed,arm=arm) for j,seed in enumerate(SEEDS)
            for arm in ARMS[j:]+ARMS[:j]]

def ordered_rows(rows,seed):
    return sorted(rows,key=lambda r:hashlib.sha256(f'{seed}|{r["problem_sha256"]}'.encode()).hexdigest())

def validate(plan):
    assert plan['candidate_kind']=='history_repeatability' and plan['schedule']==schedule()
    assert plan['arms']==list(ARMS) and plan['seeds']==list(SEEDS)
    assert plan['runtime']==dict(dtype='bfloat16',max_tokens=16000,max_model_len=17920,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.94,async_scheduling=False,chunked_prefill=True,
        seed=42,temperature=.7,top_p=.95)
    prior=HERE/'first_reflection_screen100_20260914_r2/plan.json'
    assert sha(prior)==plan['prior_plan_sha256'] and plan['rows']==read(prior)['rows']
    assert len(plan['rows'])==100 and len({r['problem_sha256'] for r in plan['rows']})==100
    assert plan['run_ids']=={'screen':'cgrs_history_repeatability100x3_20260916'}
    assert plan['arm_seconds']=={'screen':900} and plan['process_seconds']=={'screen':8400,'grade':900}
    for n,h in plan['prerequisite_sha256'].items():assert sha(ROOT/n)==h,n

def cases(plan):
    validate(plan)
    return [(s['name'],'off' if s['arm']=='R' else 'negative','original14',
             'after_first_reflection' if s['arm']=='RChistory' else 'none') for s in plan['schedule']]

def point_decision(comparisons):
    # Seed is a repeat, not an additional independently sampled question.
    primary=comparisons['RChistory_vs_RC14'];baseline=comparisons['RChistory_vs_R']
    def good(c):
        return c['accuracy_delta_pp']>=-2 and c['total_change_percent']<0 and c['thinking_change_percent']<0 and c['cap_delta']<=0
    signs=all(sum(x['total_change_percent']<0 and x['thinking_change_percent']<0 for x in c['per_seed'])>=2 for c in (primary,baseline))
    return dict(point_promising=good(primary) and good(baseline) and signs,
        status_note='Exposed100 training questions; no independent confirmation or strict synergy claim. Report all seeds, intervals and failures.')
