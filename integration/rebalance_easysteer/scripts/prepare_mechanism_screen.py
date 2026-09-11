"""Freeze a new vector-only 100-question screen and untouched 200-question reserve."""
import argparse
import json
from pathlib import Path
import random
import re
import shutil
from mechanism_candidates import ROOT, BASE, require, read, save, sha, read_vector

SEED=20260913
TRAIN='sources/ReBalance/Data/Math_Train/test.jsonl'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def norm(text):return re.sub(r'\s+','',text)


def select(train,forbidden,excluded):
    seen={norm(text) for text in forbidden}|{norm(train[i]['problem']) for i in excluded}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:
            eligible.append(i);seen.add(text)
    selected=random.Random(SEED).sample(eligible,300)
    return selected[:100],selected[100:],len(eligible)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bundle=args.output.resolve();cpu=args.cpu.resolve()
    require(not bundle.exists(),'Existing bundle is immutable')
    report=read(cpu/'summary.json');require(report['status']=='completed_cpu_diagnostic','CPU audit incomplete')
    candidates=[name for name in ('min_displacement','orthogonal_mean') if report['vectors'][name]['status']=='ready_for_fixed_gpu_screen']
    require(bool(candidates),'No vector candidate passed')
    config=ROOT/BASE/'configs'
    prior=read(config/'repeat_validation100_20260911.json')['plan']
    first=read(config/'first_step_validation100_20260910.json')
    balanced=read(config/'question_balanced_validation100_20260911.json')['selection']
    train=rows(ROOT/TRAIN)
    require(len(train)==7500 and sha(ROOT/TRAIN,source=True)==prior['train_sha256']==balanced['train_sha256'],'Training source changed')
    exclusions=dict(calibration=prior['excluded_calibration_indices'],first_step=first['validation_train_indices'],
                    repeat_gate=prior['train_indices'],question_balanced=balanced['train_indices'])
    require([len(set(exclusions[key])) for key in exclusions]==[500,100,100,100],'Prior index counts changed')
    test_paths=[f'sources/ReBalance/Data/{name}/test.jsonl' for name in ('Math_Math500','Math_GSM8K')]
    tests={name:rows(ROOT/name) for name in test_paths}
    excluded=set().union(*(set(values) for values in exclusions.values()))
    selected,reserve,eligible=select(train,[r['problem'] for group in tests.values() for r in group],excluded)
    overlap={}
    for name,indices in [('screen',selected),('confirmation',reserve)]:
        normalized={norm(train[i]['problem']) for i in indices}
        require(len(normalized)==len(indices),'Duplicate selected prompt')
        for old,values in exclusions.items():overlap[name+'_vs_'+old]=len(set(indices)&set(values))
        for old,values in tests.items():overlap[name+'_vs_'+Path(old).parent.name]=len(normalized&{norm(r['problem']) for r in values})
    overlap['screen_vs_confirmation']=len(set(selected)&set(reserve))
    require(not any(overlap.values()),'Selection overlap')
    bundle.mkdir(parents=True)
    (bundle/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n*.npy binary\n',encoding='utf-8',newline='\n')
    for name,indices in [('screen100.jsonl',selected),('confirmation200.jsonl',reserve)]:
        (bundle/name).write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices),encoding='utf-8',newline='\n')
    backup=ROOT/read(config/'mechanism_cpu_plan_20260911.json')['inputs']['backup']
    arms={}
    for name,source in [('original_dynamic',backup)]+[(name,cpu/name) for name in candidates]:
        target=bundle/'assets'/name;target.mkdir(parents=True)
        for filename in ('auto_vector.pt','fit.json'):shutil.copyfile(source/filename,target/filename)
        read_vector(target/'auto_vector.pt',backup/'auto_vector.pt')
        fit=read(target/'fit.json')
        require(fit['parameters']==read(backup/'fit.json')['parameters'],'Controller was refitted')
        arms[name]=dict(directory='assets/'+name,vector_sha256=sha(target/'auto_vector.pt'),fit_sha256=sha(target/'fit.json'))
    code=dict(read(config/'final_results_20260909.json')['source_sha256'])
    for relative in ('eval/rebalance_dynamic_eval.py','scripts/mechanism_candidates.py','scripts/prepare_mechanism_screen.py',
                     'scripts/run_mechanism_screen.py','scripts/grade_mechanism_screen.py'):
        code[BASE+relative]=sha(ROOT/BASE/relative,source=True)
    # A very conservative UTF-8 byte bound for this byte-level BPE plus fixed
    # prompt text. Actual AutoTokenizer encoding is checked before model load.
    max_problem_bytes=max(len(train[i]['problem'].encode('utf-8')) for i in selected+reserve)
    require(max_problem_bytes+2048+16000<=32768,'Conservative context bound exceeded')
    plan=dict(status='prepared_not_run',scope='New 1.5B MATH training vector screen, not MATH-500',
        count=100,new_answers_planned=100*len(arms),selection_seed=SEED,train_indices=selected,
        confirmation=dict(count=200,train_indices=reserve,status='reserved_not_authorized',dataset_sha256=sha(bundle/'confirmation200.jsonl')),
        exclusions=exclusions,eligible_count=eligible,overlap_checks=overlap,train_sha256=sha(ROOT/TRAIN,source=True),
        dataset_sha256=sha(bundle/'screen100.jsonl'),test_prompt_source_sha256={name:sha(ROOT/name,source=True) for name in tests},
        input_bound=dict(max_problem_utf8_bytes=max_problem_bytes,prompt_allowance=2048,context_capacity=32768,
                         actual_tokenizer_check='Pending existing server CPU runtime; runs before any model load'),
        model='/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B',model_files_sha256=first['model_verified_against_freeze'],
        decoder_output_layer=20,dynamic_parameters=read(backup/'fit.json')['parameters'],arms=arms,run_order=list(arms),
        runtime=dict(max_tokens=16000,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
                     temperature=.7,top_p=.95,seed=42,async_scheduling=True,chunked_prefill=False,group_timeout_seconds=600),
        source_sha256=code,source_hash_format='SHA256 after CRLF to LF normalization',cpu_summary_sha256=sha(cpu/'summary.json'),
        cpu_plan_sha256=sha(config/'mechanism_cpu_plan_20260911.json',source=True),
        timing='Fresh process per arm; pure generation excludes startup and grading. Same runtime for all 3 groups; fixed order is not a repeated speed benchmark.',
        assessment=read(config/'mechanism_cpu_plan_20260911.json')['planned_generation_screen'],
        stop_conditions=['Any asset/source/model hash mismatch stops before generation.',
                         'Existing compute processes on the GPU stop the runner; never terminate another task.',
                         'CUDA numerical error, OOM, incomplete group, pairing failure or cap violation stops the batch and preserves partial files; no automatic retry.',
                         '600 second generation deadline per arm; 1800 second batch wall deadline including runtime preflight/model startup. No expansion.',
                         'Negative efficacy is retained and not retuned. Continue the other already-fixed vector arm if engineering validity holds.'],
        gpu_authorization='NOT granted; explicit user start/confirmation required before --execute',
        expected_gpu_minutes=[15,25],gpu_cost='Actual hourly GPU rate * elapsed hours; price and GPU type are not known while server is off.',
        default_server_output='/root/autodl-tmp/results/easysteer/mechanism_screen100_20260911',
        limitations=['No hidden-state replay or regeneration of old calibration answers.',
                     'No completed test or previous validation answers reused as the new screen.',
                     '100 paired questions are a candidate screen, not a statistical accuracy-preservation guarantee.'])
    save(bundle/'plan.json',plan)
    print(json.dumps({k:plan[k] for k in ('count','new_answers_planned','eligible_count','overlap_checks','dataset_sha256','run_order')},ensure_ascii=False))


if __name__=='__main__':main()
