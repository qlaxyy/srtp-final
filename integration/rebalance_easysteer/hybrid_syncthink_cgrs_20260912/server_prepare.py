"""Read-only remote result scan and isolated binary links. No model forward."""
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    original=Path('/root/autodl-tmp/projects/srtp-final')
    vendor=Path('sources/EasySteer/vllm-steer/vllm')
    binaries={}
    for binary in (original/vendor).rglob('*.so'):
        rel=binary.relative_to(original)
        dest=ROOT/rel
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(binary)
        binaries[str(rel)]=dict(target=str(binary),bytes=binary.stat().st_size)
    reg=json.loads((HERE/'cpu_run1/data_registry.json').read_text())
    hashes={r['problem_sha256'] for r in reg['proposed_reservations']}
    ids={r['train_index'] for r in reg['proposed_reservations']}
    engineering_ids={r['train_index'] for r in reg['proposed_reservations'] if r['purpose']=='engineering'}
    engineering_hashes={r['problem_sha256'] for r in reg['proposed_reservations'] if r['purpose']=='engineering'}
    files={};hits=[];errors=[];owned_engineering=[]
    def phash(s):return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()
    def inspect(obj,path):
        if isinstance(obj,dict):
            if obj.get('train_index') in ids or any(isinstance(obj.get(k),str) and phash(obj[k]) in hashes for k in ('problem','question')):
                record=dict(path=str(path),train_index=obj.get('train_index'),keys=sorted(obj))
                own_path='/hybrid_syncthink_cgrs_20260912/s64_screen_run2_20260913/engineering/' in str(path)
                own_question=obj.get('train_index') in engineering_ids or any(isinstance(obj.get(k),str) and phash(obj[k]) in engineering_hashes for k in ('problem','question'))
                (owned_engineering if own_path and own_question else hits).append(record)
            for v in obj.values():
                if isinstance(v,(dict,list)):inspect(v,path)
        elif isinstance(obj,list):
            for v in obj:
                if isinstance(v,(dict,list)):inspect(v,path)
    for p in Path('/root/autodl-tmp/results/easysteer').rglob('*'):
        if not p.is_file() or p.suffix not in ('.json','.jsonl') or p.name=='tokenizer.json':continue
        if any(x in ('sources','node_modules','.git') for x in p.parts):continue
        data=p.read_bytes();files[str(p)]=hashlib.sha256(data).hexdigest()
        if not data:continue
        try:
            text=data.decode('utf-8-sig')
            if p.suffix=='.jsonl':
                for s in text.splitlines():
                    if s.strip():inspect(json.loads(s),p)
            else:inspect(json.loads(text),p)
        except (ValueError,UnicodeError) as e:errors.append(dict(path=str(p),error=str(e)))
    report=dict(new_unreported_result_files=hits,parse_errors=errors,owned_engineering_exposure=owned_engineering,
                scanned_files=len(files),source_sha256=files,binary_links=binaries,
                shared_git_status=subprocess.check_output(['git','-C',str(original),'status','--porcelain'],text=True),
                shared_head=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip(),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True),
                scope='Materialized JSON/JSONL in remote EasySteer result root; local archive scan covers downloaded archive members')
    with (ROOT/'remote_inventory.json').open('x') as f:json.dump(report,f,indent=2)
    print(dict(hits=len(hits),errors=len(errors),files=len(files),apps=report['compute_apps']))
    assert not hits and not errors and not report['compute_apps'].strip()


if __name__=='__main__':main()
