"""Freeze a limited SEAL comparison, without using frozen test outputs."""
import argparse
import json
from pathlib import Path
import random
import re
import shutil
from mechanism_candidates import ROOT,BASE,read,save,sha,require
from prepare_mechanism_screen import rows,norm,TRAIN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(not out.exists(),'Immutable output exists')
    config=ROOT/BASE/'configs';prior=read(config/'feedback_controlled_screen100_20260912/plan.json');fit=read(a.assets/'fit.json')
    require(fit['decoder_output_layer']==19 and fit['static_coefficient']==1 and not fit['normalize'],'SEAL formula/layer changed')
    require(sha(a.assets/'seal_vector.pt')==fit['vector_sha256'],'SEAL asset changed')
    train=rows(ROOT/TRAIN);require(sha(ROOT/TRAIN,source=True)==prior['train_sha256'],'Training source changed')
    exclusions=dict(prior['exclusions'],feedback_screen=prior['train_indices'],feedback_reserve=prior['confirmation']['train_indices'],feedback_engineering=prior['engineering']['train_indices'])
    excluded=set().union(*(set(v) for v in exclusions.values()))
    testpaths=dict(prior['test_prompt_source_sha256']);aimepath='sources/ReBalance/Data/Math_AIME2024/test.jsonl';testpaths[aimepath]=sha(ROOT/aimepath,source=True)
    tests={name:rows(ROOT/name) for name in testpaths}
    for name,digest in testpaths.items():require(sha(ROOT/name,source=True)==digest,'Test input changed')
    aime=tests[aimepath];coverage=[]
    for row in aime:
        found=re.search(r'2024_AIME_(I|II)_Problems/Problem_(\d+)',row['url']);require(found,'Unexpected AIME source URL');coverage.append((found.group(1),int(found.group(2))))
    require(len(aime)==30 and set(coverage)=={(v,i) for v in ['I','II'] for i in range(1,16)},'Not the full30 AIME2024 inputs')
    seen={norm(r['problem']) for group in tests.values() for r in group}|{norm(train[i]['problem']) for i in excluded};eligible=[]
    for i,row in enumerate(train):
        text=norm(row['problem'])
        if i not in excluded and text not in seen:eligible.append(i);seen.add(text)
    chosen=random.Random(20260916).sample(eligible,108);engineering,screen=chosen[:8],chosen[8:]
    overlap={}
    for name,indices in [('engineering',engineering),('math_train',screen)]:
        text={norm(train[i]['problem']) for i in indices};require(len(text)==len(indices),'Duplicate prompts')
        for old,values in exclusions.items():overlap[name+'_vs_'+old]=len(set(indices)&set(values))
        for old,group in tests.items():overlap[name+'_vs_'+Path(old).parent.name]=len(text&{norm(r['problem']) for r in group})
    overlap['engineering_vs_math_train']=len(set(engineering)&set(screen))
    overlap['aime_vs_calibration']=len({norm(r['problem']) for r in aime}&{norm(train[i]['problem']) for i in exclusions['calibration']})
    require(not any(overlap.values()),'Overlap')
    out.mkdir(parents=True);(out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n*.pt binary\n',encoding='utf-8',newline='\n');datasets={}
    for name,group in [('engineering',[dict(train[i],train_index=i) for i in engineering]),('math_train',[dict(train[i],train_index=i) for i in screen]),('aime2024',aime)]:
        file=name+'.jsonl';(out/file).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in group),encoding='utf-8',newline='\n');datasets[name]=dict(file=file,count=len(group),sha256=sha(out/file))
    assets={}
    for name,folder,file,layer in [('original_dynamic',config/'feedback_controlled_screen100_20260912/assets/original_dynamic','auto_vector.pt',20),('seal',a.assets,'seal_vector.pt',19)]:
        target=out/'assets'/name;target.mkdir(parents=True)
        for f in [file,'fit.json']:shutil.copyfile(folder/f,target/f)
        assets[name]=dict(vector='assets/'+name+'/'+file,fit='assets/'+name+'/fit.json',vector_sha256=sha(target/file),fit_sha256=sha(target/'fit.json'),decoder_output_layer=layer)
    paths=set(prior['source_sha256'])|{BASE+'scripts/run_seal_comparison.py',BASE+'scripts/prepare_seal_comparison.py',BASE+'scripts/fit_seal_saved.py'}
    plan=dict(status='prepared_not_run',purpose='Bounded comparison requested by user; separate from optimization candidates. No strength/layer sweep or tuning on these outputs.',
        new_comparison_answers=390,new_short_engineering_outputs=24,selection_seed=20260916,math_train_indices=screen,engineering_indices=engineering,eligible_train_count=len(eligible),
        datasets=datasets,run_order=['unsteered','original_dynamic','seal'],assets=assets,exclusions=exclusions,overlap_checks=overlap,
        source_inputs_sha256={TRAIN:sha(ROOT/TRAIN,source=True),**testpaths},source_sha256={name:sha(ROOT/name,source=True) for name in sorted(paths)},
        model=prior['model'],model_files_sha256=prior['model_files_sha256'],dynamic_parameters=prior['dynamic_parameters'],runtime=prior['runtime'],
        engine='One engine per stage, identical declared rebalance+seal additive graph across all groups. Disable prefix caching; restore request-local marker/history state across KV eviction.',
        seal_method='Self-calibrated author-SEAL pooled E-minus-(R union T), raw vector, alpha1, generated delimiter inputs after decoder19 only within think markers; existing500 greedy calibration answers reused.',
        limitations=fit['limitations']+['AIME2024 has30 questions and one sample each, so accuracy estimates are coarse. No AIME2025/Olympiad/7B in this batch.',
            'MATH100 comes from previously ungenerated training examples and is for comparison, not parameter selection or final benchmark confirmation.'],
        smoke_output='/root/autodl-tmp/results/easysteer/seal_smoke8_20260912',comparison_output='/root/autodl-tmp/results/easysteer/seal_comparison130_20260912',
        expected_gpu_minutes=dict(smoke=[2,5],comparison=[12,25]),timeout_seconds=dict(smoke=600,comparison=2400),
        metrics=['author accuracy','mean thinking tokens','mean total tokens','caps including errors','per-question correctness/token changes','pure generation seconds'],
        stop_conditions=['All data/model/source/payload checks and constant-marker CPU tests pass before model load.',
            'Smoke unsteered vs SEAL alpha0 must have identical8 output token streams; any mismatch stops before comparison.',
            'Any failure/deadline/incomplete group keeps partials and stops; no automatic regeneration.',
            'All6 complete comparison groups are reported, regardless of quality; no post-result parameter selection or expansion.'])
    save(out/'plan.json',plan);print(json.dumps(dict(status=plan['status'],count=130,arms=3,engineering=8,plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
