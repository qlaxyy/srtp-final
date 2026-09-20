"""Select a fresh, non-overlapping fixed100 proposal; never infer GPU permission."""
import hashlib
import json
from pathlib import Path
import subprocess

from engineering import ROOT, HERE, read, save, sha
from reconcile_screen import phash, scan


def main():
    out=HERE/'first_reflection_screen100_20260914'
    assert not out.exists(), 'Never replace a published selection'
    trees=[Path(s[9:]) for s in subprocess.check_output(
        ['git','-C',str(ROOT),'worktree','list','--porcelain'],text=True).splitlines()
        if s.startswith('worktree ')]
    roots=[base/'integration/rebalance_easysteer'/sub for base in trees
           for sub in ('configs','hybrid_syncthink_cgrs_20260912')
           if (base/'integration/rebalance_easysteer'/sub).is_dir()]
    source=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    initial=read(Path('E:/srtp/srtp-final/integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json'))
    excluded=set().union(*(set(v) for v in initial['exclusions'].values()))
    for stages in initial['splits'].values():
        for ids in stages.values():excluded.update(ids)
    blocked={phash(train[i]['problem']) for i in excluded}
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
    paths += [ROOT/f'sources/ReBalance/Data/{d}/test.jsonl'
              for d in ('Math_Math500','Math_GSM8K','Math_Olympiad')]
    files={}
    for p in paths:
        files[str(p)]=sha(p)
        if p.suffix=='.json':inspect(read(p))
        else:
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():inspect(json.loads(line))
    available=[];seen=set(blocked)
    for i,r in enumerate(train):
        h=phash(r['problem'])
        if h in seen or r['answer'] is None:continue
        seen.add(h);available.append(dict(r,train_index=i,problem_sha256=h,split='train',dataset='math_train'))
    salt='hybrid_syncthink_cgrs_20260912|first_reflection_screen100_v1|'
    available.sort(key=lambda r:hashlib.sha256((salt+r['problem_sha256']).encode()).hexdigest())
    assert len(available)>=100
    rows=[dict(r,dataset_index=i,purpose='proposed_first_reflection_screen') for i,r in enumerate(available[:100])]
    out.mkdir();save(out/'rows.json',rows)
    audit=scan(rows,roots,[out.resolve()]);audit['rows_sha256']=sha(out/'rows.json')
    save(out/'local_reconciliation.json',audit);assert audit['passed']
    proposal=dict(status='proposed_not_frozen_remote_reconciliation_pending',
        normalization='NFKC/remove all Unicode whitespace/SHA256 UTF8',selection_salt=salt,
        source_sha256=sha(source),available=len(available),scanned_sha256=files,
        observed_commits={str(w):subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip() for w in trees},
        reservations=[{k:r[k] for k in ('train_index','problem_sha256','purpose')} for r in rows],
        confirmation200_used=False,rows_sha256=sha(out/'rows.json'),
        limitations=['All local worktree registries scanned; remote reconciliation still required before freeze.',
          'Any collision stops this fixed proposal; no silent replacement. No new confirmation set claimed.'])
    save(out/'data_proposal.json',proposal)
    prior=read(HERE/'first_reflection_20260914/plan.json')
    old=read(HERE/'narrow8_7b_screen100_20260914/plan.json')
    evidence_dir=HERE/'first_reflection_20260914/results'
    mechanism={n:h for n,h in prior['source_sha256'].items()
               if not n.startswith(HERE.relative_to(ROOT).as_posix()) or Path(n).name in ('adapter.py','replay_adapter.py','policy.py')}
    for name,digest in mechanism.items():assert sha(ROOT/name,True)==digest,name
    plan=dict(schema=1,candidate_kind='first_reflection_screen',phase='proposed_training_screen',
        arms=['R','RC14','RChistory'],primary_candidate='RChistory',rows=rows,
        reserved_rows_sha256=sha(out/'rows.json'),data_proposal_sha256=sha(out/'data_proposal.json'),
        assets=prior['assets'],runtime=prior['runtime'],source_sha256=dict(prior['source_sha256']),
        gpu_authorized=False,data_reconciled=False,
        engineering_evidence=dict(plan_sha256=sha(HERE/'first_reflection_20260914/plan.json'),
            gate_sha256=sha(evidence_dir/'engineering_gate.json'),unchanged_mechanism_sources=mechanism,
            execution_commit='6243c3c29b1432ad4106fe781fbb6e14e0fd137f',
            result='8x6 cap512 passed; candidate equals R8/8 and differs from14 on2/8; not compression evidence'),
        run_ids=dict(screen='cgrs_first_reflection_7b_screen100_20260914'),
        arm_seconds=dict(screen=900),process_seconds=dict(screen=3000,grade=300),
        estimates=dict(minutes=[15,30],generation_answers=300,max_generated_tokens=4800000,
            basis='Previous7B new100x3 generation1329.5s plus startup and grading; tails uncertain'),
        decision=dict(old['decision'],criterion='RChistory correct>=RC14; loss vs R<=2pp; both mean lengths below R; caps<=both; accuracy improves over RC14 OR both lengths improve over RC14'),
        stop=['Any receipt/source/asset/data mismatch; no GPU loading before reconciliation',
            'Native tests fail, OOM, exception, arm900s or process3000s timeout',
            'Save partials and failures; no automatic retry, row replacement, parameter sweep or expansion'],
        limitations=old['limitations'][:])
    plan['limitations']=[s.replace('RC8','RChistory').replace('isolates vocabulary','isolates history gate') for s in plan['limitations']]
    plan['decision']['confirmation']='Existing200 remain untouched; any independent confirmation needs new batch coordination and authorization.'
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    prefix=HERE.relative_to(ROOT).as_posix();rel=out.relative_to(ROOT).as_posix()
    root='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    gate=(evidence_dir/'engineering_gate.json').relative_to(ROOT).as_posix()
    plan['commands']=dict(
        screen=f'timeout 3000s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {prefix}/narrow_runner.py --plan {rel}/plan.json --receipt {rel}/execution_receipt.json --phase screen --engineering-gate {gate} --output-root {root} --gpu-authorized',
        grade=f'timeout 300s /root/autodl-tmp/venvs/rebalance/bin/python {prefix}/narrow_grade.py --output {root}/{plan["run_ids"]["screen"]}')
    save(out/'plan.json',plan)
    save(out/'receipt_template.json',dict(gpu_authorized=False,plan_sha256=sha(out/'plan.json'),
        data_reconciled=False,unregistered_claims_checked=False,authorized_phases=['screen'],
        local_reconciliation=dict(path=None,sha256=None),remote_reconciliation=dict(path=None,sha256=None)))
    print(json.dumps(dict(rows=100,available=len(available),scanned=len(files),hits=len(audit['hits']),plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':main()
