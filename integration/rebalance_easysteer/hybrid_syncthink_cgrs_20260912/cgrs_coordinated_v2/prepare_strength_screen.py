"""Local-only new training screen proposal; no data freeze or GPU entry."""
import hashlib
import json
from pathlib import Path
import subprocess
from engineering import ROOT,HERE,read,save,sha
from reconcile_screen import phash,scan


def main():
    out=HERE/'strength_screen100_proposal_20260916'
    assert not out.exists()
    train_path=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train=[json.loads(x) for x in train_path.read_text(encoding='utf-8').splitlines()]
    trees=[Path(x[9:]) for x in subprocess.check_output(['git','-C',str(ROOT),'worktree','list','--porcelain'],text=True).splitlines() if x.startswith('worktree ')]
    roots=[]
    for tree in trees:
        for sub in ('configs','hybrid_syncthink_cgrs_20260912'):
            p=tree/'integration/rebalance_easysteer'/sub
            if p.is_dir():roots.append(p)
    roots.extend(p for p in Path('E:/srtp').glob('B-*') if p.is_dir())
    extra=Path('E:/srtp/srtp-final/.codex_work/local_prepared_draft01_20260912')
    if extra.is_dir():roots.append(extra)
    initial=read(Path('E:/srtp/srtp-final/integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json'))
    excluded=set().union(*(set(v) for v in initial['exclusions'].values()))
    for stages in initial['splits'].values():
        for ids in stages.values():excluded.update(ids)
    blocked={phash(train[i]['problem']) for i in excluded};files={};seen=set();errors=[]
    def inspect(o):
        if isinstance(o,dict):
            for k in ('problem','question'):
                if isinstance(o.get(k),str):blocked.add(phash(o[k]))
            for k in ('problem_sha256','normalized_prompt_sha256'):
                if isinstance(o.get(k),str):blocked.add(o[k])
            for k in ('train_index','train_idx'):
                if type(o.get(k)) is int and 0<=o[k]<len(train):blocked.add(phash(train[o[k]]['problem']))
            for v in o.values():
                if isinstance(v,(dict,list)):inspect(v)
        elif isinstance(o,list):
            for v in o:inspect(v)
    paths=[p for folder in roots for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl') and p.name!='gsm8k_source_train.jsonl']
    paths += [ROOT/f'sources/ReBalance/Data/{d}/test.jsonl' for d in ('Math_Math500','Math_GSM8K','Math_Olympiad')]
    for p in paths:
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();files[str(p)]=h
        if h in seen:continue
        seen.add(h)
        try:
            if p.suffix=='.json':inspect(json.loads(raw.decode('utf-8-sig')))
            else:
                for line in raw.decode('utf-8-sig').splitlines():
                    if line.strip():inspect(json.loads(line))
        except (ValueError,UnicodeError) as exc:errors.append(dict(path=str(p),error=repr(exc)))
    out.mkdir();save(out/'scan_receipt.json',dict(files=files,errors=errors,roots=list(map(str,roots)),initial_excluded_ids=sorted(excluded),train_sha256=sha(train_path)))
    assert not errors,errors
    available=[];unique=set(blocked)
    for i,r in enumerate(train):
        h=phash(r['problem'])
        if h in unique or r['answer'] is None:continue
        unique.add(h);available.append(dict(r,train_index=i,problem_sha256=h,split='train',dataset='math_train'))
    salt='hybrid_syncthink_cgrs_20260912|strength_screen100_v1|'
    available.sort(key=lambda r:hashlib.sha256((salt+r['problem_sha256']).encode()).hexdigest())
    assert len(available)>=100
    rows=[dict(r,dataset_index=i,purpose='proposed_strength_screen_not_independent_confirmation') for i,r in enumerate(available[:100])]
    save(out/'rows.json',rows)
    audit=scan(rows,roots,[out.resolve()]);save(out/'local_reconciliation.json',audit);assert audit['passed']
    save(out/'proposal.json',dict(status='proposed_not_frozen',model='DeepSeek-R1-Distill-Qwen-1.5B',unique_questions=100,
        arms=['R','RC14','RCconstant','RCscaled'],seed=42,max_new_tokens=16000,
        available=len(available),salt=salt,rows_sha256=sha(out/'rows.json'),source_sha256=sha(train_path),
        reservations=[{k:r[k] for k in ('train_index','problem_sha256','purpose')} for r in rows],
        remote_reconciliation_complete=False,gpu_authorized=False,
        limitations=['Local registered claims and explicitly included raw folders only. Executor must reconcile current remote and unregistered usage before freezing.','Old200 confirmation reservation preserved; not used or newly reclassified.','No outcome-based question selection.'] ))
    print(json.dumps(dict(available=len(available),proposed=100,scanned_files=len(files),unique_contents=len(seen),local_conflicts=len(audit['hits']))))


if __name__=='__main__':main()
