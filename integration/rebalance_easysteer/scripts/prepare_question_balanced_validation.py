"""Freeze fresh training questions and both calibration assets before generation."""
import argparse
import json
from pathlib import Path
import random
from prepare_repeat_validation import ROOT,BASE,TRAIN,TESTS,rows,norm,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True,help='Local downloaded CPU fit assets')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    prior_path=ROOT/BASE/'configs/repeat_validation100_20260911/plan.json'
    prior=json.loads(prior_path.read_text())
    train=rows(ROOT/TRAIN)
    if len(train)!=7500 or sha(ROOT/TRAIN,source=True)!=prior['train_sha256']:
        raise ValueError('Training source changed')
    calibration=prior['excluded_calibration_indices']
    previous=prior['excluded_previous_validation_indices']+prior['train_indices']
    excluded=set(calibration+previous)
    tests={name:rows(ROOT/f'sources/ReBalance/Data/{name}/test.jsonl') for name in TESTS}
    for name in TESTS:
        if sha(ROOT/f'sources/ReBalance/Data/{name}/test.jsonl',source=True)!=prior['test_prompt_source_sha256'][name]:
            raise ValueError('Held-out prompts changed')
    forbidden={norm(train[i]['problem']) for i in excluded}
    forbidden.update(norm(r['problem']) for g in tests.values() for r in g)
    seen=set(forbidden); eligible=[]
    for i,r in enumerate(train):
        n=norm(r['problem'])
        if i not in excluded and n not in seen:
            eligible.append(i); seen.add(n)
    seed=20260912
    selected=random.Random(seed).sample(eligible,100)
    prompts={norm(train[i]['problem']) for i in selected}
    overlaps={name:len(prompts & {norm(r['problem']) for r in g}) for name,g in tests.items()}
    overlaps.update(calibration=len(set(selected)&set(calibration)),previous_validation=len(set(selected)&set(previous)))
    if any(overlaps.values()) or len(prompts)!=100 or prompts & forbidden:
        raise ValueError('Question exclusion failed')
    fit=json.loads((args.candidate/'fit.json').read_text())
    audit=json.loads((args.candidate/'fit_audit.json').read_text())
    if (fit.get('method_variant')!='question-balanced-class-prototypes-v1'
        or not audit['original_vector_reproduced_bit_exact']
        or sha(args.candidate/'fit.json')!=audit['candidate_fit_sha256']
        or sha(args.candidate/'auto_vector.pt')!=fit['vector_sha256']):
        raise ValueError('Candidate fit audit mismatch')
    code={name:sha(ROOT/name,source=True) for name in prior['source_sha256']}
    for name in ('calibrate_question_balanced.py','run_question_balanced_validation.py','prepare_question_balanced_validation.py'):
        code[BASE+'scripts/'+name]=sha(ROOT/BASE/'scripts'/name,source=True)
    original={k:prior[k] for k in ('assets','vector_sha256','fit_sha256','dynamic_parameters')}
    candidate=dict(assets='/root/autodl-tmp/results/easysteer/question_balanced_fit_1p5b_20260911',
        vector_sha256=fit['vector_sha256'],fit_sha256=sha(args.candidate/'fit.json'),dynamic_parameters=fit['parameters'])
    plan=dict(status='prepared_not_run',scope='Fresh 1.5B MATH training validation, not MATH-500',
        count=100,new_answers_planned=200,selection_seed=seed,train_indices=selected,
        excluded_calibration_indices=calibration,excluded_previous_validation_indices=previous,
        eligible_count=len(eligible),overlap_checks=overlaps,unique_prompts=100,
        prior_plan_sha256=sha(prior_path),train_sha256=prior['train_sha256'],
        test_prompt_source_sha256=prior['test_prompt_source_sha256'],
        model=prior['model'],model_files_sha256=prior['model_files_sha256'],decoder_output_layer=20,
        runtime=prior['runtime'],calibrations=dict(original_dynamic=original,question_balanced=candidate),
        run_order=['original_dynamic','question_balanced'],source_sha256=code,
        comparison='Only class mean/variance weighting and its derived vector/crossing target change. Same layer, labels, quantiles and controller formula. No repetition detector or first-prompt injection.',
        assessment='One fixed pair. Include all wrong/capped answers. Report correctness, total/thinking tokens, caps, flips, generation time, preemptions. No retuning on this set; no automatic expansion.',
        timing='Separate processes in one fixed order; generation excludes model loading. This is not a repeated speed benchmark.',
        authorization='User authorized CPU fitting followed directly by GPU on reopened SSH20403.',
        source_hash_format='SHA256 CRLF normalized to LF; dataset and artifact hashes use raw bytes')
    args.output.mkdir(parents=True)
    (args.output/'.gitattributes').write_text('* text eol=lf\n',newline='\n')
    dataset=args.output/'validation100.jsonl'
    dataset.write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in selected),encoding='utf-8',newline='\n')
    plan['dataset_sha256']=sha(dataset)
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:plan[k] for k in ('count','eligible_count','overlap_checks','dataset_sha256')}))


if __name__=='__main__': main()
