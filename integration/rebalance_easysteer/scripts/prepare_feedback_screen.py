"""Freeze feedback engineering checks, a new100 pair and an unseen200 reserve."""
import argparse
import json
from pathlib import Path
import random
import shutil
from mechanism_candidates import ROOT, BASE, read, save, sha, require
from prepare_mechanism_screen import rows,norm,TRAIN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(not out.exists(),'Bundle exists')
    cpu=read(a.cpu/'latent_feedback_clip/fit.json');require(cpu['passes_cpu_gate'],'CPU mechanism gate failed')
    config=ROOT/BASE/'configs';prior=read(config/'confidence_only_screen100_20260912/plan.json')
    train=rows(ROOT/TRAIN);require(sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Training data changed')
    exclusions=dict(prior['exclusions'],confidence_screen=prior['train_indices'],confidence_reserve=prior['confirmation']['train_indices'])
    excluded=set().union(*(set(v) for v in exclusions.values()))
    tests={name:rows(ROOT/name) for name in prior['test_prompt_source_sha256']}
    for name,digest in prior['test_prompt_source_sha256'].items():require(sha(ROOT/name,source=True)==digest,'Test source changed')
    seen={norm(r['problem']) for group in tests.values() for r in group}|{norm(train[i]['problem']) for i in excluded}
    eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:eligible.append(i);seen.add(text)
    selected=random.Random(20260915).sample(eligible,308);smoke,screen,reserve=selected[:8],selected[8:108],selected[108:]
    overlap={}
    for name,indices in [('smoke',smoke),('screen',screen),('confirmation',reserve)]:
        text={norm(train[i]['problem']) for i in indices};require(len(text)==len(indices),'Duplicate prompts')
        for old,values in exclusions.items():overlap[name+'_vs_'+old]=len(set(indices)&set(values))
        for old,group in tests.items():overlap[name+'_vs_'+Path(old).parent.name]=len(text&{norm(r['problem']) for r in group})
    overlap.update(smoke_vs_screen=len(set(smoke)&set(screen)),smoke_vs_confirmation=len(set(smoke)&set(reserve)),screen_vs_confirmation=len(set(screen)&set(reserve)))
    require(not any(overlap.values()),'Overlap')
    out.mkdir(parents=True);(out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n*.npy binary\n',encoding='utf-8',newline='\n')
    for file,indices in [('smoke8.jsonl',smoke),('screen100.jsonl',screen),('confirmation200.jsonl',reserve)]:
        (out/file).write_text(''.join(json.dumps(dict(train[i],train_index=i),ensure_ascii=False)+'\n' for i in indices),encoding='utf-8',newline='\n')
    backup=ROOT/read(config/'overnight_research_20260912.json')['first_investigation']['inputs']['backup'];arms={}
    for name in ['original_dynamic','latent_feedback_clip']:
        folder=out/'assets'/name;folder.mkdir(parents=True)
        for file in ['auto_vector.pt','fit.json']:shutil.copyfile(backup/file,folder/file)
        arms[name]=dict(directory='assets/'+name,vector_sha256=sha(folder/'auto_vector.pt'),fit_sha256=sha(folder/'fit.json'))
    folder=out/'assets/latent_feedback_clip';shutil.copyfile(a.cpu/'latent_feedback_clip/readout.npy',folder/'readout.npy')
    require(sha(folder/'readout.npy')==cpu['readout_sha256'],'Readout changed')
    feedback=dict(readout_file='readout.npy',readout_sha256=cpu['readout_sha256'],vector_sha256=arms['original_dynamic']['vector_sha256'],
        decoder_output_layer=20,hidden_state_index=21,negative_centroid_score=cpu['negative_centroid_score'],
        formula='c_eff=c for c>=0; otherwise max(c,-max(0,(w.h-w.mu_under)/(w.d))); FP32 readout uses the actual BF16 injected direction response.',
        dtype='BF16 direction/activation/coefficient and FP32 readout/center; rounding means the linear-state bound is approximate.',
        source_cpu_fit_sha256=sha(a.cpu/'latent_feedback_clip/fit.json'),counter_scope='Actual coefficient changes; application executions include any KV replay. FP32 amplitude totals are diagnostic only.')
    save(folder/'feedback.json',feedback);arms['latent_feedback_clip'].update(feedback_config='assets/latent_feedback_clip/feedback.json',feedback_config_sha256=sha(folder/'feedback.json'))
    paths=set(read(config/'final_results_20260909.json')['source_sha256'])
    paths.update(prior['source_sha256'])
    paths.update(BASE+x for x in ['scripts/prepare_feedback_screen.py','scripts/check_feedback_graph.py','scripts/run_feedback_engineering.py'])
    paths.update('sources/EasySteer/vllm-steer/'+x for x in ['vllm/steer_vectors/graph_kernels.py','vllm/steer_vectors/payloads.py','vllm/steer_vectors/algorithms/rebalance.py','vllm/steer_vectors/algorithms/__init__.py','tests/steer_vectors/test_rebalance.py'])
    code={name:sha(ROOT/name,source=True) for name in sorted(paths)}
    plan=dict(status='prepared_not_run',stage='screen',scope='One-factor online latent-feedback clip with original vector/controller',
        count=100,new_answers_planned=200,dataset_file='screen100.jsonl',dataset_sha256=sha(out/'screen100.jsonl'),selection_seed=20260915,
        train_indices=screen,eligible_count=len(eligible),exclusions=exclusions,overlap_checks=overlap,
        confirmation=dict(count=200,train_indices=reserve,dataset_file='confirmation200.jsonl',dataset_sha256=sha(out/'confirmation200.jsonl'),status='untouched_conditional_reserve'),
        engineering=dict(count=8,train_indices=smoke,dataset_file='smoke8.jsonl',dataset_sha256=sha(out/'smoke8.jsonl'),max_tokens=256,
            order=['original_dynamic','feedback_disabled','feedback_enabled'],new_short_outputs=24,
            purpose='Compiled/CUDA graph behavior and exact disabled-payload token identity; short outputs are engineering data, never efficacy evidence.',
            kernel_checks='128 synthetic BF16 rows; disabled-vs-old compiled exact; enabled compiled-vs-eager absolute error <= (abs(hidden)+abs(compiled-hidden)+1)/128; captured replay exact for mutable masks and enabled flags.',
            stop='Any mismatch, graph failure, incomplete group or deadline stops before100-screen. Enabled short outputs may differ; no grading or parameter selection from these8.',
            expected_gpu_minutes=[3,10],batch_timeout_seconds=1800,group_timeout_seconds=600),
        train_sha256=prior['train_sha256'],test_prompt_source_sha256=prior['test_prompt_source_sha256'],model=prior['model'],model_files_sha256=prior['model_files_sha256'],
        decoder_output_layer=20,dynamic_parameters=prior['dynamic_parameters'],runtime=prior['runtime'],arms=arms,run_order=list(arms),source_sha256=code,
        cpu_summary_sha256=sha(a.cpu/'summary.json'),hypothesis=read(config/'mechanism_followups_20260912.json')['candidates'][1],
        assessment='Both thinking and total mean tokens reduce >=5%, correct count not lower, caps not higher. No threshold/strength search. Any confirmation uses unchanged candidate on sealed200.',
        expected_gpu_minutes=[6,15],batch_timeout_seconds=1800,
        stop_conditions=['CPU and CUDA math checks and compiled/short-generation engineering checks must pass.','Any model/source/input/asset mismatch or another task: stop before model load.','Any incomplete or failed arm: preserve progress and stop, no automatic retry.','Failed efficacy criterion ends this fixed candidate without tuning or expanding.'],
        default_server_output='/root/autodl-tmp/results/easysteer/latent_feedback_screen100_20260912',authorization='User8-hour ongoing GPU research; bounded engineering24 short outputs then200screen answers.',
        limitations=['Clipping a fitted linear readout does not prove semantic safety.','Application counters add measured runtime cost and may count replay executions.','Engineering cases, screening cases and confirmation cases are all disjoint.','A100-pair is a screen, not a narrow accuracy guarantee.'])
    save(out/'plan.json',plan)
    print(json.dumps(dict(status=plan['status'],screen=100,reserve=200,engineering=8,overlap_checks=overlap)))


if __name__=='__main__':main()
