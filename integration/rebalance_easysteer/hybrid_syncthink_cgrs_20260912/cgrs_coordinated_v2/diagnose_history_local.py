"""Read-only trajectory/progress audit. No inference, grading or parameter search."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
from engineering import ROOT,HERE,read,save,sha

def first_diff(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b)) if len(a)!=len(b) else None

def category(diff,new_first,old_first=-1):
    if diff is None:return 'identical'
    starts=[x for x in (new_first,old_first) if x>=0]
    if not starts:return 'different_without_either_intervention'
    boundary=min(starts)
    if diff<boundary:return 'before_either_intervention'
    if diff==boundary:return 'at_first_intervention'
    return 'after_first_intervention'

def main():
    out=HERE/'local_diagnosis_20260916';out.mkdir(exist_ok=False)
    manifest={};summary={};details={}
    def checked(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected:assert digest==expected,str(p)
        manifest[str(p)]=digest
        return read(p)
    Rpaths={
      ('1p5b','math_test'):'auto_code_v2_500_20260908/math500_eval.json',
      ('1p5b','gsm8k_test'):'auto_code_v2_500_20260908/gsm8k_eval.json',
      ('7b','math_test'):'qwen7b_final_20260909/final_math500_completed_20260909/math500_eval.json',
      ('7b','gsm8k_test'):'qwen7b_validation/auto_code_v2_qwen7b_20260908/formal_kv_replay/gsm8k_eval.json'}
    for model,raw,oldbase,oldid in [
      ('1p5b',Path('E:/srtp/B-history-full1p5b_0915/results'),'cgrs_v2_full_run1_verified','cgrs_coordinated_v2_full_math500_gsm1319_run1_20260913'),
      ('7b',Path('E:/srtp/B-history-full7b_0914/results'),'cgrs_v2_7b_run1_verified','cgrs_coordinated_v2_full_7b_math500_gsm1319_run1_20260913')]:
        plan=checked(raw/'resolved_plan.json');analysis=checked(raw/'analysis.json')
        old=ROOT/'.codex_work'/oldbase/'results/easysteer/hybrid_syncthink_cgrs_20260912'/oldid
        oldanalysis=checked(old/'analysis.json',plan['rc14_reference']['analysis_sha256'])
        for role,sub in plan['datasets'].items():
            key=model+'_'+role;folder=raw/role/'RChistory'
            new=checked(folder/'result.json',analysis['datasets'][role]['result_sha256'])
            rc=checked(old/role/'RCnegative/result.json',plan['rc14_reference']['results'][role]['sha256'])
            frozen=checked(Path('E:/srtp/srtp-final/.codex_work')/Rpaths[model,role],plan['frozen_benchmarks'][role]['artifacts']['evaluation_sha256'])
            hist=checked(raw/'historical_reference.json')
            rows=[]
            for i,(r,c,b) in enumerate(zip(new['records'],rc['records'],frozen['rebalance_dynamic']['records'])):
                assert r['dataset_index']==c['dataset_index']==b['dataset_index']==i
                assert r['problem_sha256']==c['problem_sha256']==sub['rows'][i]['problem_sha256']
                assert r['problem']==b['problem']
                en=new['events'][r['request_id']];ec=rc['events'][c['request_id']]
                dn=first_diff(r['token_ids'],b['token_ids']);dc=first_diff(r['token_ids'],c['token_ids'])
                label=analysis['datasets'][role]['grades'][i]['correct']
                row=dict(dataset_index=i,problem_sha256=r['problem_sha256'],first_change=en['first_change'],
                    first_reflection=en['first_reflection'],rc14_first_change=ec['first_change'],
                    token_delta_vs_RC14=r['tokens']-c['tokens'],correct=label,
                    tokens=r['tokens'],rc14_tokens=c['tokens'],capped=r['finish_reason']=='length',
                    rc14_capped=c['finish_reason']=='length',vs={})
                for arm,diff,start in [('R',dn,-1),('RC14',dc,ec['first_change'])]:
                    refcorrect=hist[role]['groups'][arm]['records'][i]['correct']
                    row['vs'][arm]=dict(first_difference=diff,category=category(diff,en['first_change'],start),
                        correct_delta=int(label)-int(refcorrect))
                rows.append(row)
            assert len(rows)==len(sub['rows'])
            result=dict(n=len(rows),prefix={},accuracy_transitions={},token_delta_by_correctness={},
                no_intervention_questions=sum(r['first_change']<0 for r in rows),
                old_R_protocol={k:v for k,v in frozen['protocol'].items() if k!='dynamic_params'},
                old_R_commit=frozen['provenance']['commit'])
            for arm in ('R','RC14'):
                result['prefix'][arm]=dict(Counter(r['vs'][arm]['category'] for r in rows))
                result['accuracy_transitions'][arm]={str(d):dict(Counter(r['vs'][arm]['category'] for r in rows if r['vs'][arm]['correct_delta']==d)) for d in (-1,1)}
            for d in (-1,0,1):
                subset=[r for r in rows if r['vs']['RC14']['correct_delta']==d]
                result['token_delta_by_correctness'][str(d)]=dict(n=len(subset),total_delta=sum(r['token_delta_vs_RC14'] for r in subset))
            result['caps']=dict(new=sum(r['capped'] for r in rows),old=sum(r['rc14_capped'] for r in rows),
                new_only=[r['dataset_index'] for r in rows if r['capped'] and not r['rc14_capped']],
                resolved=[r['dataset_index'] for r in rows if not r['capped'] and r['rc14_capped']])
            result['lengths']={name:dict(total=sum(r['tokens'] for r in data['records']),
                maximum=max(r['tokens'] for r in data['records']),
                top5_total=sum(sorted((r['tokens'] for r in data['records']),reverse=True)[:5]),
                aggregate_tokens_per_second=sum(r['tokens'] for r in data['records'])/data['generation_seconds'],
                generation_seconds=data['generation_seconds'],checkpoint_io_seconds=data['checkpoint_io_seconds']) for name,data in [('history',new),('RC14',rc)]}
            snapshots=[checked(p) for p in sorted(folder.glob('progress_*.json'))]
            tail={}
            for remaining in (1,8,32):
                matches=[s for s in snapshots if s['waiting']==0 and 0<s['running']<=remaining]
                if matches:
                    s=matches[0];tail[str(remaining)]=dict(first_observed_elapsed=s['elapsed_seconds'],
                        completed=s['completed'],running=s['running'],remaining_seconds_lower_bound=new['generation_seconds']-s['elapsed_seconds'])
                else:tail[str(remaining)]=None
            result['tail_observations_30s_resolution']=tail
            result['largest_token_increases']=[dict(dataset_index=r['dataset_index'],delta=r['token_delta_vs_RC14']) for r in sorted(rows,key=lambda r:r['token_delta_vs_RC14'],reverse=True)[:5]]
            result['largest_token_decreases']=[dict(dataset_index=r['dataset_index'],delta=r['token_delta_vs_RC14']) for r in sorted(rows,key=lambda r:r['token_delta_vs_RC14'])[:5]]
            summary[key]=result;details[key]=rows
    save(out/'summary.json',dict(datasets=summary,limitations=[
        'Earlier divergence than either intervention excludes direct local lexical action as its cause; does not identify batch numerical or state error cause.',
        'After-intervention divergence is compatible with an effect, not proof of beneficial causality.',
        'Historical R protocols differ; RC14 comparison uses earliest intervention of BOTH trajectories.',
        'No logits, per-step batch membership, CUDA kernel timings or per-question finish timestamps were saved.',
        '30s snapshots give observed lower bounds for tails, not exact duration or counterfactual savings.',
        'All subgroup and extreme-row analyses are post hoc descriptive; no retuning or new efficacy test.']))
    save(out/'per_question.json',details);save(out/'input_hashes.json',manifest)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
