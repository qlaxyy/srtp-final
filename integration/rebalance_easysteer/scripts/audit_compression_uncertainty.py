"""Descriptive paired uncertainty and outcome-transition accounting; CPU only."""
import argparse
import time
from pathlib import Path
import numpy as np
from mechanism_candidates import ROOT,BASE,read,save,sha,require


BATCHES = [
 ('position','control_point_screen100_all/control_point_screen100_resumed_20260912',
  [('original_dynamic','predecessor_delimiter','prespecified_primary'),
   ('matched_first','predecessor_delimiter','prespecified_second_control'),
   ('original_dynamic','matched_first','posthoc_signal_not_primary')]),
 ('confidence_only','confidence_only_screen100_20260912_all/confidence_only_screen100_20260912',
  [('original_dynamic','confidence_only_direction','prespecified_primary')]),
 ('matched_first_confirmation','matched_first_confirmation200_20260912_all/matched_first_confirmation200_20260912',
  [('original_dynamic','matched_first','independent_confirmation_of_posthoc_signal')]),
 ('feedback','feedback_controlled_all_20260912/feedback_controlled_screen100_20260912',
  [('original_dynamic','latent_feedback_clip','prespecified_primary'),
   ('feedback_disabled','latent_feedback_clip','prespecified_second_control')]),
 ('positive_branch','positive_branch_all_20260912/positive_branch_screen100_20260912',
  [('original_dynamic','negative_only_dynamic','prespecified_primary')]),
 ('progression','trajectory_progress_screen100_20260912_all/trajectory_progress_screen100_20260912',
  [('original_dynamic','within_trajectory_progress','prespecified_primary')]),
 ('radial','radial_completed_20260912/radial_fp32_screen100_20260912',
  [('original_dynamic','radial_restore','prespecified_primary'),
   ('radial_disabled','radial_restore','prespecified_second_control')]),
]


def transitions(labels,delta):
    result={}
    for label in sorted(set(labels)):
        mask=np.array([value==label for value in labels])
        result[label]=dict(count=int(mask.sum()),total_token_delta=int(delta[mask].sum()),
            contribution_to_overall_mean_total_delta=float(delta[mask].sum()/len(delta)),
            conditional_mean_total_delta=float(delta[mask].mean()))
    require(sum(v['total_token_delta'] for v in result.values())==int(delta.sum()),'Decomposition failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();require(not a.output.exists(),'Immutable analysis exists')
    started=time.perf_counter();base=ROOT/'.codex_work/overnight_research_20260912'
    plan_path=ROOT/BASE/'configs/compression_uncertainty_audit_20260912.json'
    result=[];inputs={};unique_answers=0;unique_generation_seconds=0.
    for batch_id,(batch,relative,pairs) in enumerate(BATCHES):
        folder=base/relative;analysis=read(folder/'analysis.json')
        require(analysis['status']=='completed','Incomplete batch')
        for name,digest in analysis['source_sha256'].items():
            require(sha(folder/name)==digest,'Previously analyzed source changed: '+name)
            inputs[str((folder/name).relative_to(ROOT))]=digest
        inputs[str((folder/'analysis.json').relative_to(ROOT))]=sha(folder/'analysis.json')
        names=sorted({name for first,second,_ in pairs for name in [first,second]})
        raw={name:read(folder/(name+'.json'))['rebalance_dynamic'] for name in names}
        grades={name:read(folder/(name+'.author.json')) for name in names}
        arrays={};n=len(raw[names[0]]['records'])
        for name in names:
            records=raw[name]['records'];scored=grades[name]['records']
            require(len(records)==len(scored)==n and n in (100,200),'Wrong question count')
            require(grades[name]['input_sha256']==sha(folder/(name+'.json')),'Unpaired grade')
            for row in records:
                ids=row['token_ids'];thinking=ids.index(151649) if 151649 in ids else len(ids)
                require(row['tokens']==len(ids)<=16000 and row['thinking_tokens']==thinking,'Token accounting mismatch')
            arrays[name]=dict(total=np.array([r['tokens'] for r in records]),
                thinking=np.array([r['thinking_tokens'] for r in records]),
                correct=np.array([int(r['correct']) for r in scored]),
                capped=np.array([r['finish_reason']=='length' or r['tokens']==16000 for r in records]))
            require(int(arrays[name]['correct'].sum())==grades[name]['correct'],'Grade total mismatch')
            unique_answers+=n;unique_generation_seconds+=raw[name]['summary']['generation_seconds']
        index=np.random.default_rng(20260923+batch_id).integers(0,n,size=(10000,n))
        for first,second,role in pairs:
            x,y=arrays[first],arrays[second]
            require([r['problem'] for r in raw[first]['records']]==[r['problem'] for r in raw[second]['records']],'Question mismatch')
            require([r['train_index'] for r in grades[first]['records']]==[r['train_index'] for r in grades[second]['records']],'Grade order mismatch')
            metrics={}
            for metric in ['thinking','total']:
                ratios=100*(y[metric][index].mean(axis=1)/x[metric][index].mean(axis=1)-1)
                metrics[metric]=dict(control_mean=float(x[metric].mean()),candidate_mean=float(y[metric].mean()),
                    percent_change=float(100*(y[metric].mean()/x[metric].mean()-1)),
                    paired_bootstrap_95_percent_interval=np.quantile(ratios,[.025,.975]).tolist())
            diff=y['correct']-x['correct']
            metrics['accuracy']=dict(control_correct=int(x['correct'].sum()),candidate_correct=int(y['correct'].sum()),
                difference_percentage_points=float(100*diff.mean()),
                paired_bootstrap_95_pp_interval=np.quantile(100*diff[index].mean(axis=1),[.025,.975]).tolist(),
                improved=int((diff>0).sum()),degraded=int((diff<0).sum()))
            cap_labels=[f'{int(a)}_to_{int(b)}' for a,b in zip(x['capped'],y['capped'])]
            correct_labels=[f'{a}_to_{b}' for a,b in zip(x['correct'],y['correct'])]
            result.append(dict(batch=batch,control=first,candidate=second,role=role,count=n,metrics=metrics,
                control_caps=int(x['capped'].sum()),candidate_caps=int(y['capped'].sum()),
                cap_transitions=transitions(cap_labels,y['total']-x['total']),
                correctness_transitions=transitions(correct_labels,y['total']-x['total']),
                control_generation_seconds=raw[first]['summary']['generation_seconds'],
                candidate_generation_seconds=raw[second]['summary']['generation_seconds'],
                limitation='Outcome-conditioned decomposition and single-seed bootstrap; no changed stopping decision or deployable gate.'))
    save(a.output,dict(status='completed_posthoc_CPU_description',plan_sha256=sha(plan_path,source=True),
        script_sha256=sha(Path(__file__),source=True),input_sha256=inputs,comparisons=result,
        unique_generated_answers_in_selected_batches=unique_answers,
        summed_pure_generation_seconds_without_counting_shared_controls_twice=unique_generation_seconds,
        excluded_cost='SEAL/AIME, saved-state replays, failed wrong-layer outputs, engineering, prefix probes and startup are deliberately outside this seven-batch generation sum; it is not total GPU billed time.',
        cpu_seconds=time.perf_counter()-started,new_answers=0,regrading=0,GPU_calls=0))
    print(dict(comparisons=len(result),unique_answers=unique_answers,selected_batch_pure_seconds=unique_generation_seconds))
    for row in result:
        print(row['batch'],row['control'],row['metrics']['total']['paired_bootstrap_95_percent_interval'],row['metrics']['accuracy']['paired_bootstrap_95_pp_interval'])


if __name__=='__main__':main()
