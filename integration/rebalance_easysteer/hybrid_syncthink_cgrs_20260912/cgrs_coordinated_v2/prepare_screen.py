"""Local deterministic proposal and reproducibility package; not a data freeze."""
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata

from engineering import ROOT, HERE, read, save, sha


def phash(text):
    return hashlib.sha256(''.join(unicodedata.normalize('NFKC',text).split()).encode()).hexdigest()


def main():
    out=HERE/'screen64_run1';assert not out.exists()
    peer=Path('E:/srtp/srtp-final')
    source=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    initial=read(peer/'integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json')
    excluded=set().union(*(set(v) for v in initial['exclusions'].values()))
    for stages in initial['splits'].values():
        for ids in stages.values():excluded.update(ids)
    blocked={phash(train[i]['problem']) for i in excluded}
    scanned={};errors=[]
    def inspect(obj):
        if isinstance(obj,dict):
            for key in ('problem','question'):
                if isinstance(obj.get(key),str):blocked.add(phash(obj[key]))
            for key in ('problem_sha256','normalized_prompt_sha256'):
                if isinstance(obj.get(key),str) and len(obj[key])==64:blocked.add(obj[key])
            # Conservatively exclude ID-only records too, including ambiguous dataset labels.
            for key in ('train_index','train_idx'):
                i=obj.get(key)
                if type(i) is int and 0<=i<len(train):blocked.add(phash(train[i]['problem']))
            for value in obj.values():
                if isinstance(value,(dict,list)):inspect(value)
        elif isinstance(obj,list):
            for value in obj:
                if isinstance(value,(dict,list)):inspect(value)
    worktrees=[Path(s[9:]) for s in subprocess.check_output(
        ['git','-C',str(ROOT),'worktree','list','--porcelain'],text=True).splitlines() if s.startswith('worktree ')]
    for base in worktrees:
        for folder in (base/'integration/rebalance_easysteer/configs',
                       base/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
            for p in folder.rglob('*'):
                if not p.is_file() or p.suffix not in ('.json','.jsonl') or p.name=='gsm8k_source_train.jsonl':continue
                scanned[str(p)]=sha(p)
                try:
                    if p.suffix=='.json':inspect(read(p))
                    else:
                        for line in p.read_text(encoding='utf-8-sig').splitlines():
                            if line.strip():inspect(json.loads(line))
                except (ValueError,UnicodeError) as exc:errors.append(dict(path=str(p),error=repr(exc)))
    assert not errors,errors
    for dataset in ('Math_Math500','Math_GSM8K'):
        p=ROOT/f'sources/ReBalance/Data/{dataset}/test.jsonl';scanned[str(p)]=sha(p)
        for line in p.read_text(encoding='utf8').splitlines():inspect(json.loads(line))
    eligible=[];seen=set(blocked)
    for i,row in enumerate(train):
        h=phash(row['problem'])
        if h in seen or row['answer'] is None:continue
        seen.add(h);eligible.append(dict(row,train_index=i,problem_sha256=h,split='train',dataset='math_train'))
    salt='hybrid_syncthink_cgrs_20260912|cgrs_coordinated_v2_screen64_v1|'
    eligible.sort(key=lambda r:hashlib.sha256((salt+r['problem_sha256']).encode()).hexdigest())
    assert len(eligible)>=64
    selected=eligible[:64];assert not ({r['problem_sha256'] for r in selected}&blocked)
    out.mkdir()
    save(out/'rows.json',selected)
    proposal=dict(status='proposed_not_frozen_pending_executor_remote_reconciliation',
        normalization='NFKC then remove all Unicode whitespace; SHA256 UTF8',selection_salt=salt,
        available=len(eligible),excluded_unique_hashes=len(blocked),scanned_sha256=scanned,
        parse_errors=errors,source_sha256=sha(source),rows_sha256=sha(out/'rows.json'),
        observed_commits={str(p):subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in worktrees},
        reservations=[dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],
            purpose='proposed_v2_screening_not_independent_confirmation') for r in selected],
        local_overlap_count=0,remote_reconciliation_complete=False,
        limitations=['Local config/registry scan is not proof of unregistered remote/private usage',
          'Final executor must recheck local and remote A/B claims before freezing; any new collision stops without replacing rows'])
    save(out/'data_proposal.json',proposal)
    plan=read(HERE/'engineering_plan.json')
    plan.update(run_id='cgrs_coordinated_v2_screen64_run1_20260913',phase='screen',
        arms=['U','R','Clex','RCalways','RCnegative'],rows=selected,
        primary_candidate='RCnegative',gpu_authorized=False,data_reconciled=False,
        estimated_minutes=[15,25],process_ceiling_seconds=2100,arm_ceiling_seconds=360,
        controls='U/R/Clex/RCalways factorial; RCnegative versus RCalways isolates gate; Clex not official CGRS',
        stop=['source/asset/data mismatch','native CPU failure','KV preemption','accepted-token clock mismatch',
              'timeout or exception; preserve completed partials; no automatic threshold changes or retry'],
        data_proposal_sha256=sha(out/'data_proposal.json'))
    plan['runtime']['max_tokens']=16000
    for p in HERE.glob('*.py'):plan['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p,True)
    prefix='integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
    output='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912'
    plan['command']=f'timeout 2100s /root/autodl-tmp/venvs/easysteer-vllm026/bin/python {prefix}/screen.py --plan {prefix}/screen64_run1/plan.json --receipt {prefix}/screen64_run1/execution_receipt.json --output-root {output} --gpu-authorized'
    plan['grade_command']=f'timeout 600s /root/autodl-tmp/venvs/rebalance/bin/python {prefix}/screen_grade.py --output {output}/{plan["run_id"]}'
    plan['grade_resume_command']=plan['grade_command']+' --resume'
    plan['decision_protocol']=read(HERE/'implementation_spec.json')['next_effect_design']
    plan['decision_protocol']['status']='parameters_fixed_data_proposed_not_frozen'
    plan['decision_protocol']['primary']='RCnegative vs R; RCalways is a diagnostic arm, not an outcome-selected replacement'
    plan['timing']='Report end-to-end generation, separate setup/I/O subsets and GPU sample envelope. Added probes/forwards=0. Control GPU kernels not separately timed; paired timing includes unequal output workloads.'
    plan['deployment']='Unique isolated copy of validated v2 run1; do not edit shared vLLM or old outputs. Requires updated local and remote data reconciliation receipts and fresh bounded GPU authorization before launch.'
    save(out/'plan.json',plan)
    print(json.dumps(dict(proposed=64,scanned=len(scanned),eligible=len(eligible),indices=[r['train_index'] for r in selected],plan_sha256=sha(out/'plan.json')),indent=2))


if __name__=='__main__':main()
