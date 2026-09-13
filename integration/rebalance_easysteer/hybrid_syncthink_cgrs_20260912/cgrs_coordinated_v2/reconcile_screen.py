"""Read-only cross-line registry scan; an executor supplies explicit roots."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def phash(s):
    return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()


def scan(rows, roots, exclude):
    targets={r['problem_sha256'] for r in rows};ids={r['train_index'] for r in rows}
    hits=[];errors=[];files={};seen=set()
    def inspect(o,label):
        if isinstance(o,dict):
            hashes={phash(o[k]) for k in ('problem','question') if isinstance(o.get(k),str)}
            hashes.update(o[k] for k in ('problem_sha256','normalized_prompt_sha256') if isinstance(o.get(k),str))
            matched_ids={o[k] for k in ('train_index','train_idx') if type(o.get(k)) is int and o[k] in ids}
            if hashes&targets or matched_ids:
                hits.append(dict(path=label,hashes=sorted(hashes&targets),ids=sorted(matched_ids)))
            for v in o.values():
                if isinstance(v,(dict,list)):inspect(v,label)
        elif isinstance(o,list):
            for v in o:
                if isinstance(v,(dict,list)):inspect(v,label)
    for folder in roots:
        if not folder.is_dir():
            errors.append(dict(path=str(folder),error='Missing requested registry root'));continue
        for p in folder.rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.jsonl'):continue
            if any(p.resolve().is_relative_to(e) for e in exclude) or p.name=='gsm8k_source_train.jsonl':continue
            raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();files[str(p)]=digest
            if digest in seen:continue
            seen.add(digest)
            try:
                if p.suffix=='.json':inspect(json.loads(raw.decode('utf-8-sig')),str(p))
                else:
                    for line in raw.decode('utf-8-sig').splitlines():
                        if line.strip():inspect(json.loads(line),str(p))
            except (ValueError,UnicodeError) as exc:errors.append(dict(path=str(p),error=repr(exc)))
    return dict(passed=not hits and not errors,hits=hits,errors=errors,source_sha256=files,
        unique_contents=len(seen),roots=[str(p) for p in roots],excluded_own_proposal_paths=[str(p) for p in exclude],
        limitations=['Explicit roots only; executor must include all current A/B registry locations and confirm no unregistered claims',
          'Ambiguous ID-only records are conservatively treated as conflicts'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--exclude-own-proposal',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=json.loads(args.rows.read_text(encoding='utf8'))
    result=scan(rows,args.root,[x.resolve() for x in args.exclude_own_proposal])
    result['rows_sha256']=hashlib.sha256(args.rows.read_bytes()).hexdigest()
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=result['passed'],files=len(result['source_sha256']),hits=len(result['hits']),errors=len(result['errors']))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
