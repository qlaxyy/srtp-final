"""Freeze the progression-vector screen and conditional independent reserve."""
import argparse
import json
from pathlib import Path
import random
import shutil

from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import rows, norm, TRAIN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),'Immutable bundle exists')
    cpu=read(args.cpu/'summary.json');require(cpu['passes_cpu_gate'],'CPU gate failed')
    config=ROOT/BASE/'configs';hypothesis=config/'within_trajectory_progress_20260912.json'
    require(sha(hypothesis,source=True)==cpu['plan_sha256'],'CPU plan changed')
    prior=read(config/'positive_branch_screen100_20260912/plan.json')
    exclusions=dict(prior['exclusions'],positive_branch_screen=prior['train_indices'],
                    positive_branch_reserve=prior['confirmation']['train_indices'],
                    positive_branch_engineering=prior['engineering']['train_indices'])
    excluded=set().union(*(set(v) for v in exclusions.values()))
    train=rows(ROOT/TRAIN);require(sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Train source changed')
    tests={name:rows(ROOT/name) for name in prior['test_prompt_source_sha256']}
    for name,digest in prior['test_prompt_source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Test source changed')
    seen={norm(r['problem']) for group in tests.values() for r in group}|{norm(train[i]['problem']) for i in excluded}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:eligible.append(i);seen.add(text)
    chosen=random.Random(20260921).sample(eligible,300);screen,reserve=chosen[:100],chosen[100:]
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
    backup=ROOT/read(hypothesis)['inputs']['backup'];arms={}
    for name,source in [('original_dynamic',backup),('within_trajectory_progress',args.cpu/'within_trajectory_progress')]:
        folder=out/'assets'/name;folder.mkdir(parents=True)
        for file in ['auto_vector.pt','fit.json']:shutil.copyfile(source/file,folder/file)
        arms[name]=dict(directory='assets/'+name,vector_sha256=sha(folder/'auto_vector.pt'),fit_sha256=sha(folder/'fit.json'))
    shutil.copyfile(args.cpu/'summary.json',out/'cpu_summary.json')
    paths=set(prior['source_sha256'])
    paths.update(BASE+name for name in ['eval/calibration_contract.py','scripts/fit_trajectory_progress.py','scripts/prepare_trajectory_screen.py'])
    plan=dict(status='prepared_not_run',stage='screen',scope='Fresh100 MATH training within-trajectory progression vector screen; replace only direction',
        count=100,new_answers_planned=200,dataset_file='screen100.jsonl',dataset_sha256=sha(out/'screen100.jsonl'),
        selection_seed=20260921,train_indices=screen,eligible_count=len(eligible),exclusions=exclusions,overlap_checks=overlap,
        confirmation=dict(count=200,train_indices=reserve,dataset_file='confirmation200.jsonl',dataset_sha256=sha(out/'confirmation200.jsonl'),status='untouched_conditional_reserve'),
        train_sha256=prior['train_sha256'],test_prompt_source_sha256=prior['test_prompt_source_sha256'],
        model=prior['model'],model_files_sha256=prior['model_files_sha256'],decoder_output_layer=20,
        dynamic_parameters=prior['dynamic_parameters'],runtime=prior['runtime'],arms=arms,run_order=list(arms),
        source_sha256={name:sha(ROOT/name,source=True) for name in sorted(paths)},cpu_summary_sha256=sha(out/'cpu_summary.json'),
        hypothesis=read(hypothesis),assessment='Both mean thinking and total token reduction >=5%; correct count not lower; caps not higher. One screen per arm. Only if passing, unchanged candidate vs fresh original on separately reserved200. No retuning.',
        expected_gpu_minutes=[6,12],batch_timeout_seconds=1500,
        stop_conditions=['Input/model/source mismatch or another GPU task: stop before model load.','Any generation error, incomplete arm, invalid pairing, cap or deadline: preserve partial outputs and stop without automatic retry.','Failed efficacy gate: stop candidate and leave200 reserve untouched.'],
        default_server_output='/root/autodl-tmp/results/easysteer/trajectory_progress_screen100_20260912',
        authorization='User explicit at-least-eight-hour research request with GPU open; bounded1.5B paired screen and conditional200 confirmation.',
        engineering='Same verified additive graph, payload and controller. Actual-parser no-model preflight checks both new assets and version-independent layer metadata; no redundant short generation required.',
        limitations=read(hypothesis)['limitations']+['100-question screening is selection evidence, not final accuracy-preservation evidence.'])
    save(out/'plan.json',plan)
    print(json.dumps(dict(count=100,new_answers=200,reserve=200,eligible=len(eligible),plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
