"""Prepare the authorized independent MATH200/GSM8K200 batch, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import zipfile
from prepare_execution import phash, save, digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE/'expanded_20260913'
GSM_COMMIT='3101c7d5072418e28b9008a6636bde82a006892c'
URL=f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM_COMMIT}/grade_school_math/data/train.jsonl'


def local():
    DATA.mkdir(exist_ok=True)
    source_path=DATA/'gsm8k_source_train.jsonl'
    source=source_path.read_bytes() if source_path.exists() else urllib.request.urlopen(URL,timeout=60).read()
    if not source_path.exists():source_path.write_bytes(source)
    assert not (DATA/'local_audit.json').exists(), 'Completed audit cannot be overwritten'
    gsm=[json.loads(s) for s in source.decode().splitlines()]
    assert len(gsm)==7473
    reg=json.loads((HERE/'cpu_run1/data_registry.json').read_text(encoding='utf-8'))
    reserved=[r for r in reg['proposed_reservations'] if r['purpose']=='confirmation']
    assert len(reserved)==200
    math_source=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    train=[json.loads(s) for s in math_source.read_text(encoding='utf-8').splitlines()]
    math=[dict(train[r['train_index']],train_index=r['train_index'],problem_sha256=r['problem_sha256'],dataset='math_train') for r in reserved]
    assert all(phash(r['problem'])==r['problem_sha256'] for r in math)
    target_hashes={r['problem_sha256'] for r in math}
    target_ids={r['train_index'] for r in math}
    blocked=set();hits=[];errors=[];scans={};seen=set()
    def inspect(obj,label):
        if isinstance(obj,dict):
            local_hashes={phash(obj[k]) for k in ('problem','question') if isinstance(obj.get(k),str)}
            blocked.update(local_hashes)
            if obj.get('train_index') in target_ids or target_hashes & local_hashes:
                hits.append(dict(path=label,train_index=obj.get('train_index')))
            for v in obj.values():
                if isinstance(v,(dict,list)):inspect(v,label)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):inspect(v,label)
    def scan(label,data):
        h=digest(data);scans[label]=h
        if h in seen or not data:return
        seen.add(h)
        try:
            text=data.decode('utf-8-sig')
            if label.endswith('.jsonl'):
                for s in text.splitlines():
                    if s.strip():inspect(json.loads(s),label)
            else:inspect(json.loads(text),label)
        except (ValueError,UnicodeError) as e:errors.append(dict(path=label,error=str(e)))
    aroot=Path('E:/srtp/srtp-final')
    for directory in (aroot/'integration/rebalance_easysteer/configs',aroot/'.codex_work'):
        for p in directory.rglob('*'):
            if not p.is_file() or any(x in ('sources','upstream','original_snapshot','node_modules','.git') for x in p.relative_to(directory).parts):continue
            label=str(p)
            if p.name=='tokenizer.json':continue
            if p.suffix in ('.json','.jsonl'):scan(label,p.read_bytes())
            elif p.name.endswith(('.tar.gz','.tgz','.zip')):
                if p.suffix=='.zip':
                    with zipfile.ZipFile(p) as z:
                        for n in z.namelist():
                            if n.endswith(('.json','.jsonl')) and '/sources/' not in '/'+n:scan(label+'::'+n,z.read(n))
                else:
                    with tarfile.open(p) as t:
                        for m in t:
                            if m.isfile() and m.name.endswith(('.json','.jsonl')) and '/sources/' not in '/'+m.name:scan(label+'::'+m.name,t.extractfile(m).read())
    # Full upstream sources are only duplicate-exclusion pools, never outcomes.
    for p in [ROOT/'sources/ReBalance/Data'/name/'test.jsonl' for name in ('Math_Math500','Math_GSM8K')]:
        for s in p.read_text(encoding='utf-8-sig').splitlines():
            r=json.loads(s)
            for k in ('problem','question'):
                if isinstance(r.get(k),str):blocked.add(phash(r[k]))
    blocked.update(phash(r['problem']) for r in train)
    eligible={}
    for i,r in enumerate(gsm):
        h=phash(r['question'])
        if h not in blocked and h not in eligible:
            eligible[h]=dict(dataset='gsm8k_train',train_index=i,id=f'gsm8k-train-{i}',problem=r['question'],answer=r['answer'].split('####')[-1].strip().replace(',',''),cot=r['answer'],problem_sha256=h)
    chosen=sorted(eligible.values(),key=lambda r:digest(('hybrid_syncthink_cgrs_20260912|gsm8k_confirmation|'+r['problem_sha256']).encode()))[:200]
    audit=dict(A_head=subprocess.check_output(['git','-C',str(aroot),'rev-parse','HEAD'],text=True).strip(),scanned_files=len(scans),unique_contents=len(seen),source_sha256=scans,hits=hits,errors=errors,eligible_gsm8k=len(eligible),gsm8k_source=dict(url=URL,commit=GSM_COMMIT,sha256=digest(source)),normalization=reg['normalization'],scope='Current A configs/materialized results/archive members; B pre-reserved MATH200; frozen MATH500/GSM8K tests and all MATH training excluded from GSM selection by normalized hash; remote output audit required before freeze.')
    save(DATA/'local_audit.json',audit)
    assert not hits and not errors and len(chosen)==200,(len(hits),len(errors))
    for role,rows in [('math',math),('gsm8k',chosen)]:
        with (DATA/(role+'.jsonl')).open('x',encoding='utf-8',newline='\n') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    usage=[dict(dataset=r['dataset'],train_index=r['train_index'],problem_sha256=r['problem_sha256'],purpose='independent_confirmation',status='reserved_pending_remote_audit') for r in math+chosen]
    save(DATA/'data_usage.json',dict(records=usage,previous_usage='screen64_20260913/data_usage.json',normalization=reg['normalization']))
    save(DATA/'plan.json',dict(authorization='User approved MATH200 + GSM8K200 x four arms, faster configuration',run_id='s64_confirm_math200_gsm200_async_run1_20260913',roles=['math','gsm8k'],counts=200,arms=['U','R','S','RS'],async_scheduling=True,max_num_seqs=128,max_num_batched_tokens=32768,max_tokens=16000,temperature=.7,top_p=.95,seed=42,method=dict(entropy_weight=.8,pacing_cap=64),engineering='8 reused engineering prompts x pre_R/off_R/shadow_R/S/RS, 512 max new; async validation only, not independent outcomes',estimated_minutes=[20,45],whole_budget_seconds=4200,engineering_budget_seconds=300,arm_budget_seconds=900,stop='Any data/asset mismatch, other GPU work, OOM/preemption, nonfinite/clock failure, off/shadow identity failure, pretrigger divergence, missing/duplicate result, or wall budget. Preserve partial; no automatic method tuning or rerun.',primary='MATH independent confirmation; GSM8K prespecified transfer replication, reported separately, no pooled rescue',criteria='Original MATH confirmation standard unchanged: RS lower mean thinking/total than BOTH R and S, correct count >= both, caps <= both, actual force >0; BOTH total difference 97.5% two-sided bootstrap upper <0 and one-sided97.5% Clopper-Pearson upper gross single-correct/RS-wrong probability <=2%. Same criteria separately on GSM8K. 10000 paired bootstrap seed20260912. Additive interaction I=T_RS-T_R-T_S+T_U,95% upper<0 required for superadditivity; do not equate point superiority with synergy.',files={role:digest((DATA/(role+'.jsonl')).read_bytes()) for role in ('math','gsm8k')}))
    print(json.dumps({k:audit[k] for k in ('scanned_files','unique_contents','hits','errors','eligible_gsm8k')}))


def remote():
    import run_batch as rb
    from transformers import AutoTokenizer
    from rebalance_static_eval import build_prompt
    import torch,vllm
    plan=rb.read(HERE/'experiment_plan.json');exp=rb.read(DATA/'plan.json')
    for n,m in plan['assets']['model_files'].items():assert rb.sha(Path(plan['assets']['model_path'])/n)==m['sha256'],n
    for k in ('vector','fit'):assert rb.sha(plan['assets'][k]['path'])==plan['assets'][k]['sha256'],k
    rows={role:rb.dataset(role,True) for role in ('engineering','math','gsm8k')}
    for role,h in exp['files'].items():assert rb.sha(DATA/(role+'.jsonl'))==h
    hashes={r['problem_sha256'] for role in ('math','gsm8k') for r in rows[role]};hits=[];files={}
    def inspect(obj,path):
        if isinstance(obj,dict):
            if any(isinstance(obj.get(k),str) and phash(obj[k]) in hashes for k in ('problem','question')):hits.append(path)
            for v in obj.values():
                if isinstance(v,(dict,list)):inspect(v,path)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):inspect(v,path)
    for p in Path('/root/autodl-tmp/results/easysteer').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl'):
            raw=p.read_bytes();files[str(p)]=digest(raw)
            if not raw:continue
            if p.suffix=='.jsonl':
                for s in raw.decode('utf-8-sig').splitlines():
                    if s.strip():inspect(json.loads(s),str(p))
            else:inspect(json.loads(raw),str(p))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    save(ROOT/'expanded_remote_audit.json',dict(hits=hits,source_sha256=files,compute_apps=apps))
    assert not hits and not apps.strip()
    tok=AutoTokenizer.from_pretrained(plan['assets']['model_path'],local_files_only=True)
    prompts={role:[tok.encode(build_prompt(tok,r['problem'])) for r in group] for role,group in rows.items()}
    assert max(len(p) for group in prompts.values() for p in group)+16000<32768
    sources={}
    for folder in ('sources/EasySteer/vllm-steer/vllm','sources/EasySteer/easysteer','sources/ReBalance/utils','integration/rebalance_easysteer/eval','integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl'):sources[p.relative_to(ROOT).as_posix()]=rb.sha(p)
    assert rb.sha(ROOT/'baseline_model_runner.py')=='88d36451373681a3e82526ad6de69a64356818a8a8777c72d82b51ab8bebf8f5'
    sources['baseline_model_runner.py']=rb.sha(ROOT/'baseline_model_runner.py')
    save(ROOT/'resolved_expanded.json',dict(expansion=exp,run_id=exp['run_id'],data_frozen=True,source_sha256=sources,plan_sha256=rb.sha(HERE/'experiment_plan.json'),assets=plan['assets'],prompts=prompts,output='/root/autodl-tmp/results/easysteer/'+plan['namespace']+'/'+exp['run_id'],environment=dict(build_tools=rb.check_build_tools(),torch=torch.__version__,vllm=vllm.__version__),deployment=rb.read(ROOT/'deployment_identity.json'),local_audit_sha256=rb.sha(DATA/'local_audit.json'),remote_audit_sha256=rb.sha(ROOT/'expanded_remote_audit.json')))
    print('READY',rb.sha(ROOT/'resolved_expanded.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote',action='store_true');a=p.parse_args()
    remote() if a.remote else local()
