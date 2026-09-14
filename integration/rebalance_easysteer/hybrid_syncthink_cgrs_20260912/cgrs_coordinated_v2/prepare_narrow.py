"""One-shot local proposal; does not freeze data or authorize any generation."""
import hashlib
import json
from pathlib import Path
import subprocess
from engineering import ROOT,HERE,read,save,sha
from prepare_screen import phash
from reconcile_screen import scan


def main():
    out=HERE/'narrow8_7b_screen100_20260914'
    assert not out.exists(),'Never regenerate or replace a published proposal'
    worktrees=[Path(s[9:]) for s in subprocess.check_output(
        ['git','-C',str(ROOT),'worktree','list','--porcelain'],text=True).splitlines() if s.startswith('worktree ')]
    source=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    initial=read(Path('E:/srtp/srtp-final/integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json'))
    excluded=set().union(*(set(v) for v in initial['exclusions'].values()))
    for stages in initial['splits'].values():
        for ids in stages.values():excluded.update(ids)
    blocked={phash(train[i]['problem']) for i in excluded};files={};errors=[]
    roots=[];missing=[]
    for base in worktrees:
        assert base.is_dir()
        for sub in ('configs','hybrid_syncthink_cgrs_20260912'):
            folder=base/'integration/rebalance_easysteer'/sub
            if folder.is_dir():roots.append(folder)
            else:missing.append(str(folder))
    def inspect(o):
        if isinstance(o,dict):
            for k in ('problem','question'):
                if isinstance(o.get(k),str):blocked.add(phash(o[k]))
            for k in ('problem_sha256','normalized_prompt_sha256'):
                if isinstance(o.get(k),str):blocked.add(o[k])
            for k in ('train_index','train_idx'):
                if type(o.get(k)) is int and 0<=o[k]<len(train):blocked.add(phash(train[o[k]]['problem']))
            for v in o.values():
                if isinstance(v,(list,dict)):inspect(v)
        elif isinstance(o,list):
            for v in o:inspect(v)
    paths=[p for folder in roots for p in folder.rglob('*') if p.is_file()
        and p.suffix in ('.json','.jsonl') and p.name!='gsm8k_source_train.jsonl']
    paths += [ROOT/f'sources/ReBalance/Data/{d}/test.jsonl' for d in ('Math_Math500','Math_GSM8K','Math_Olympiad')]
    for p in paths:
        files[str(p)]=sha(p)
        try:
            if p.suffix=='.json':inspect(read(p))
            else:
                for line in p.read_text(encoding='utf-8-sig').splitlines():
                    if line.strip():inspect(json.loads(line))
        except (ValueError,UnicodeError) as e:errors.append(dict(path=str(p),error=repr(e)))
    assert not errors,errors
    available=[];seen=set(blocked)
    for i,r in enumerate(train):
        h=phash(r['problem'])
        if h in seen or r['answer'] is None:continue
        seen.add(h);available.append(dict(r,train_index=i,problem_sha256=h,split='train',dataset='math_train'))
    salt='hybrid_syncthink_cgrs_20260912|narrow8_7b_screen100_confirm200_v1|'
    available.sort(key=lambda r:hashlib.sha256((salt+r['problem_sha256']).encode()).hexdigest())
    assert len(available)>=300
    screen=[dict(r,dataset_index=i,purpose='proposed_screen') for i,r in enumerate(available[:100])]
    confirm=[dict(r,dataset_index=i,purpose='reserved_independent_confirmation_not_authorized') for i,r in enumerate(available[100:300])]
    out.mkdir();save(out/'screen_rows.json',screen);save(out/'confirmation_rows.json',confirm)
    save(out/'reserved_rows.json',screen+confirm)
    audit=scan(screen+confirm,roots,[out.resolve()]);audit['rows_sha256']=sha(out/'reserved_rows.json')
    save(out/'local_reconciliation.json',audit);assert audit['passed']
    proposal=dict(status='proposed_not_frozen',source_sha256=sha(source),normalization='NFKC/remove all Unicode whitespace/SHA256 UTF8',
        selection_salt=salt,available=len(available),scanned_sha256=files,parse_errors=errors,
        absent_registry_directories=missing,excluded_initial_ids=sorted(excluded),
        observed_commits={str(w):subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip() for w in worktrees},
        rows_sha256=audit['rows_sha256'],remote_reconciliation_complete=False,
        reservations=[{k:r[k] for k in ('train_index','problem_sha256','purpose')} for r in screen+confirm],
        limitations=['Local registered claims only; executor must reconcile all current local/remote A/B claims and unregistered usage',
            'Any collision stops this proposal; no silent replacement. Confirmation200 cannot be used by either line for development'])
    save(out/'data_proposal.json',proposal)
    prior=read(HERE/'full_7b_run1/plan.json')
    plan=dict(schema=1,phase='proposed_training_screen',arms=['R','RC14','RC8'],rows=screen,
        engineering_rows=prior['engineering_rows'],speed_rows=read(HERE/'screen64_run1/rows.json'),
        reserved_rows_sha256=audit['rows_sha256'],data_proposal_sha256=sha(out/'data_proposal.json'),
        assets=prior['assets'],runtime=prior['runtime'],source_sha256=prior['source_sha256'],
        gpu_authorized=False,data_reconciled=False,primary_candidate='RC8',
        speed_profiles={'current32':{'max_num_seqs':32},'candidate48':{'max_num_seqs':48}},
        run_ids={phase:'cgrs_narrow8_7b_'+name+'_20260914' for phase,name in
                 [('engineering','engineering'),('screen','screen100'),('speed','speed64')]},
        arm_seconds={'engineering':120,'screen':900,'speed':300},
        process_seconds={'engineering':900,'screen':3000,'speed_each_profile':600,'grade':300},
        estimates=dict(method_batch_minutes=[15,30],method_hard_ceiling_seconds=4200,
            speed_separate_minutes=[3,8],speed_combined_ceiling_seconds=1200,
            basis='Rough previous7B throughput scaling plus startup; 100-question batches and tails may differ'),
        generation_counts=dict(engineering=48,screen=300,speed_optional=128,confirmation_authorized=0),
        maximum_generated_tokens=dict(engineering=12288,screen=4800000,speed_optional=131072),
        decision=dict(status='fixed_before_generation',criterion='RC8 correct>=RC14; loss vs R<=2pp; both mean lengths below R; caps<=both; accuracy improves over RC14 OR both lengths improve over RC14',
            tradeoff='Accuracy recovery may lose some RC14 compression; report as tradeoff, not Pareto dominance',
            inference='10000 paired bootstrap, seed20260913; point2pp bound separate from nominal95% lower CI>=-2pp',
            confirmation='200 reserved only; new authorization and reconciliation required; no automatic retuning or expansion'),
        speed_decision='Report same64 prompts/cap1024, output throughput, elapsed/startup, discarded KV and replays. 48 is exploratory; no automatic promotion to long-output runs, no accuracy or strict equivalence claim.',
        stop=['Data/source/asset mismatch or missing receipts','Native CPU or forced-replay equivalence failure',
            'No RC8 activation in engineering (cannot claim coverage)','OOM, clock mismatch, exception, arm or process timeout',
            'Preserve failures and completed/partial outputs; no automatic retry, row replacement or retuning'],
        limitations=['Prior MATH500 diagnosis motivated hypothesis; no new test-set tuning',
            'R/RC14/RC8 isolates vocabulary; no U or lexical-only arm, so cannot establish factorial synergy',
            'Fixed arm order and seed; paired question CIs omit runtime/sampling variability',
            'Zero probes and auxiliary forwards; kernel overhead not separately profiled; KV replay costs included',
            'Short speed window cannot establish sustained long-context throughput'])
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    prefix=HERE.relative_to(ROOT).as_posix();rel=out.relative_to(ROOT).as_posix()
    output='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    base=f'/root/autodl-tmp/venvs/easysteer-vllm026/bin/python {prefix}/narrow_runner.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --output-root {output} --gpu-authorized'
    plan['commands']={
        'working_directory':'Validated isolated B checkout; never switch A/shared checkout',
        'engineering':f'timeout 900s {base} --phase engineering',
        'screen':f'timeout 3000s {base} --phase screen --engineering-gate {output}/{plan["run_ids"]["engineering"]}/engineering_gate.json',
        'grade':f'timeout 300s /root/autodl-tmp/venvs/rebalance/bin/python {prefix}/narrow_grade.py --output {output}/{plan["run_ids"]["screen"]}',
        'speed32':f'timeout 600s {base} --phase speed --profile current32',
        'speed48':f'timeout 600s {base} --phase speed --profile candidate48',
        'reconcile_template':f'python {prefix}/reconcile_screen.py --rows {rel}/reserved_rows.json --root <each-current-A-and-B-registry-root> --exclude-own-proposal {rel} --output <new-local-or-remote-reconciliation.json>'}
    save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,plan_sha256=sha(out/'plan.json'),
        data_reconciled=False,unregistered_claims_checked=False,authorized_phases=[],authorized_speed_profiles=[],
        local_reconciliation=dict(path=None,sha256=None),remote_reconciliation=dict(path=None,sha256=None),
        note='Fill only after actual coordination and bounded user GPU authorization; local proposal audit is not a live freeze receipt'))
    print(json.dumps(dict(screen=100,reserved=200,eligible=len(available),scanned=len(files),local_overlap=len(audit['hits']),
                         plan_sha256=sha(out/'plan.json')),indent=2))


if __name__=='__main__':main()
