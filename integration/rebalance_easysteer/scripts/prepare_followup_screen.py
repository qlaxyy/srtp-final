"""Freeze the confidence-only candidate's new screen and untouched reserve."""
import argparse
import json
from pathlib import Path
import random
import shutil
from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import rows,norm,TRAIN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(not out.exists(),'Immutable bundle exists')
    report=read(a.cpu/'summary.json')['confidence_only_direction'];require(report['passes_cpu_gate'],'CPU gate failed')
    config=ROOT/BASE/'configs';prior=read(config/'control_point_screen100_20260912/plan.json')
    train=rows(ROOT/TRAIN);require(sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Train source changed')
    exclusions=dict(prior['exclusions'],position_screen=prior['train_indices'],position_reserve=prior['confirmation']['train_indices'])
    excluded=set().union(*(set(v) for v in exclusions.values()))
    tests={name:rows(ROOT/name) for name in prior['test_prompt_source_sha256']}
    for name,digest in prior['test_prompt_source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Test source changed')
    seen={norm(row['problem']) for group in tests.values() for row in group}|{norm(train[i]['problem']) for i in excluded}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:eligible.append(i);seen.add(text)
    selected=random.Random(20260914).sample(eligible,300);screen,reserve=selected[:100],selected[100:]
    overlap={}
    for name,indices in [('screen',screen),('confirmation',reserve)]:
        text={norm(train[i]['problem']) for i in indices};require(len(text)==len(indices),'Duplicate prompts')
        for old,values in exclusions.items():overlap[name+'_vs_'+old]=len(set(indices)&set(values))
        for old,group in tests.items():overlap[name+'_vs_'+Path(old).parent.name]=len(text&{norm(r['problem']) for r in group})
    overlap['screen_vs_confirmation']=len(set(screen)&set(reserve));require(not any(overlap.values()),'Input overlap')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n',encoding='utf-8',newline='\n')
    for name,indices in [('screen100.jsonl',screen),('confirmation200.jsonl',reserve)]:
        (out/name).write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices),encoding='utf-8',newline='\n')
    backup=ROOT/read(config/'overnight_research_20260912.json')['first_investigation']['inputs']['backup']
    arms={}
    for name,source in [('original_dynamic',backup),('confidence_only_direction',a.cpu/'confidence_only_direction')]:
        folder=out/'assets'/name;folder.mkdir(parents=True)
        for filename in ('auto_vector.pt','fit.json'):shutil.copyfile(source/filename,folder/filename)
        arms[name]=dict(directory='assets/'+name,vector_sha256=sha(folder/'auto_vector.pt'),fit_sha256=sha(folder/'fit.json'))
    code=dict(read(config/'final_results_20260909.json')['source_sha256'])
    for relative in ['eval/rebalance_dynamic_eval.py','scripts/mechanism_candidates.py','scripts/run_mechanism_screen.py',
                     'scripts/run_vector_batch.py','scripts/grade_vector_batch.py','scripts/prepare_followup_screen.py']:
        code[BASE+relative]=sha(ROOT/BASE/relative,source=True)
    plan=dict(status='prepared_not_run',stage='screen',scope='Fresh MATH training confidence-only direction screen; one vector-label factor',
        count=100,new_answers_planned=200,dataset_file='screen100.jsonl',dataset_sha256=sha(out/'screen100.jsonl'),
        selection_seed=20260914,train_indices=screen,eligible_count=len(eligible),exclusions=exclusions,overlap_checks=overlap,
        confirmation=dict(count=200,train_indices=reserve,dataset_file='confirmation200.jsonl',dataset_sha256=sha(out/'confirmation200.jsonl'),status='untouched_conditional_reserve'),
        train_sha256=prior['train_sha256'],test_prompt_source_sha256=prior['test_prompt_source_sha256'],
        model=prior['model'],model_files_sha256=prior['model_files_sha256'],decoder_output_layer=20,
        dynamic_parameters=prior['dynamic_parameters'],runtime=prior['runtime'],arms=arms,run_order=list(arms),
        source_sha256=code,cpu_summary_sha256=sha(a.cpu/'summary.json'),
        hypothesis=read(config/'mechanism_followups_20260912.json')['candidates'][0],
        assessment='Both mean thinking and total token reduction >=5%; correct count not lower; caps not higher. One screen per arm. If passing, unchanged candidate vs fresh original on the separately reserved200, no retuning.',
        expected_gpu_minutes=[6,12],batch_timeout_seconds=1500,
        stop_conditions=['Asset/model/source mismatch or another GPU task: stop before model load.','Actual evaluator layer and controller preflight must pass.','Any generation error, incomplete arm, cap/pairing violation: preserve outputs and stop; no automatic retry.','Negative efficacy: retain and stop without tuning or expanding this candidate.'],
        default_server_output='/root/autodl-tmp/results/easysteer/confidence_only_screen100_20260912',
        authorization='Explicit user eight-hour active GPU research request; this fixed pair and conditional200 are within its scope.',
        limitations=['Original mixed-label calibration is replaced by nonlexical low-vs-high confidence labels; this is a hypothesis, not a faithful author-method reproduction.','Legacy rebalance_source_layer output field is hardcoded19 for this fit version; actual parsed output layer20 and asset hidden_state_index21 are checked.','100-question screening cannot establish a narrow accuracy-preservation guarantee.'])
    save(out/'plan.json',plan)
    print(json.dumps(dict(count=100,new_answers=200,eligible=len(eligible),overlap=overlap,dataset_sha256=plan['dataset_sha256'])))


if __name__=='__main__':main()
