"""Freeze a held-out vector confirmation; explicitly label post-hoc hypotheses."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from mechanism_candidates import ROOT, BASE, read, save, sha, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen-bundle',type=Path,required=True)
    p.add_argument('--screen-results',type=Path,required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--posthoc-rationale')
    a=p.parse_args();source=a.screen_bundle.resolve();out=a.output.resolve()
    require(not out.exists(),'Confirmation already prepared')
    parent=read(source/'plan.json');analysis=read(a.screen_results/'analysis.json')
    ledger=read(a.screen_results/'run_ledger.json')
    require(analysis['status']=='completed' and ledger['plan_sha256']==sha(source/'plan.json'),'Unverified screen')
    require(a.candidate in parent['arms'] and a.candidate!='original_dynamic','Wrong candidate')
    if a.posthoc_rationale:
        # This path is allowed only for the observed matched-support control.
        # It preserves the primary negative decision; it does not relabel it.
        require(a.candidate=='matched_first' and analysis['candidate_for_separate_confirmation'] is None,'Unexpected post-hoc scope')
        require(analysis['comparisons']['matched_first']['passes_fixed_screen'],'No exploratory signal')
        stage='posthoc_confirmation'
    else:
        require(analysis.get('eligible_for_confirmation') and analysis['candidate']==a.candidate,'Primary screen did not pass')
        stage='confirmation'
    reserve=parent['confirmation'];indices=reserve['train_indices']
    dataset=source/reserve.get('dataset_file','confirmation200.jsonl')
    require(reserve['count']==len(indices)==200 and sha(dataset)==reserve['dataset_sha256'],'Reserve changed')
    require(not set(indices)&set(parent['train_indices']),'Screen/reserve overlap')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n',encoding='utf-8',newline='\n')
    shutil.copyfile(dataset,out/'confirmation200.jsonl')
    arms={}
    for name in ['original_dynamic',a.candidate]:
        asset=parent['arms'][name];folder=out/'assets'/name;folder.mkdir(parents=True)
        for filename,key in [('auto_vector.pt','vector_sha256'),('fit.json','fit_sha256')]:
            src=source/asset['directory']/filename;require(sha(src)==asset[key],'Screen asset changed')
            shutil.copyfile(src,folder/filename)
        arms[name]=dict(asset,directory='assets/'+name)
    # Same generation implementation as the completed screen; new orchestration
    # adds explicit layer selection and validated 200-question accounting.
    code=read(ROOT/BASE/'configs/confidence_only_screen100_20260912/plan.json')['source_sha256'].copy()
    code[BASE+'scripts/prepare_vector_confirmation.py']=sha(Path(__file__),source=True)
    runtime=dict(parent['runtime'],group_timeout_seconds=900)
    overlap={k:v for k,v in parent['overlap_checks'].items() if k.startswith('confirmation_')}
    overlap['confirmation_vs_parent_screen']=len(set(indices)&set(parent['train_indices']))
    other=read(ROOT/BASE/'configs/confidence_only_screen100_20260912/plan.json')
    if a.candidate=='matched_first':
        overlap['confirmation_vs_confidence_screen']=len(set(indices)&set(other['train_indices']))
        overlap['confirmation_vs_confidence_reserve']=len(set(indices)&set(other['confirmation']['train_indices']))
    require(not any(overlap.values()),'Confirmation overlap')
    plan=dict(status='prepared_not_run',stage=stage,scope='Independent200 MATH train confirmation of '+a.candidate,
        count=200,new_answers_planned=400,dataset_file='confirmation200.jsonl',dataset_sha256=sha(out/'confirmation200.jsonl'),
        train_indices=indices,selection_seed=parent['selection_seed'],original_reserve_plan_sha256=sha(source/'plan.json'),
        original_reserve_fixed_before_screen=True,overlap_checks=overlap,
        model=parent['model'],model_files_sha256=parent['model_files_sha256'],decoder_output_layer=20,
        dynamic_parameters=parent['dynamic_parameters'],runtime=runtime,arms=arms,run_order=list(arms),
        source_sha256=code,parent_analysis_sha256=sha(a.screen_results/'analysis.json'),
        primary_screen_decision_preserved=analysis.get('candidate_for_separate_confirmation',analysis.get('eligible_for_confirmation')),
        posthoc_rationale=a.posthoc_rationale,
        assessment='Confirm only if BOTH mean thinking and total tokens decrease >=5%, correct count is not lower, and caps are not higher. No parameter changes, alternative reserve subsets, or repeated runs. Report paired differences and uncertainty even if passing.',
        expected_gpu_minutes=[8,18],batch_timeout_seconds=2400,
        stop_conditions=['Any source/input/model/asset mismatch or another GPU task stops before model load.','Actual evaluator parser/spec must select layer20 and unchanged dynamic parameters.','Any runtime failure or incomplete group stops; preserve completed and partial output.','Failure of confirmation criterion ends this exact candidate. No automatic7B or full-test expansion.'],
        default_server_output='/root/autodl-tmp/results/easysteer/'+a.candidate+'_confirmation200_20260912',
        authorization='User explicitly requested eight-hour ongoing GPU research and independent held-out confirmation for promising observations; no server-mode changes.',
        limitations=['Post-hoc origin must be reported when present; the primary position-alignment candidate remains negative.','A200-question independent pair provides confirmation evidence, not a narrow accuracy non-inferiority guarantee.','The new900-second per-group deadline only increases allowed completion time; maximum generated tokens remains16000.'])
    save(out/'plan.json',plan)
    print(json.dumps(dict(status=plan['status'],stage=stage,candidate=a.candidate,count=200,new_answers=400,
        dataset_sha256=plan['dataset_sha256'],posthoc=bool(a.posthoc_rationale),overlap_checks=overlap)))


if __name__=='__main__':main()
