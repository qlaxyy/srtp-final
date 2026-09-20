"""Hash-only exposure scan for external benchmark IDs; never treat them as MATH train indices."""
import argparse
import json
from pathlib import Path
from engineering import read, save, sha
from prepare_screen import phash


def scan(rows, roots, excluded):
    targets={r['problem_sha256'] for r in rows};hits=[];errors=[];files={}
    def inspect(o,path):
        if isinstance(o,dict):
            hs={phash(o[k]) for k in ('problem','question') if isinstance(o.get(k),str)}
            hs.update(o[k] for k in ('problem_sha256','normalized_prompt_sha256') if isinstance(o.get(k),str))
            if hs&targets:hits.append(dict(path=path,hashes=sorted(hs&targets)))
            for x in o.values():
                if isinstance(x,(dict,list)):inspect(x,path)
        elif isinstance(o,list):
            for x in o:
                if isinstance(x,(dict,list)):inspect(x,path)
    for root in roots:
        if not root.is_dir():errors.append(dict(path=str(root),error='Missing registry root'));continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.jsonl'):continue
            if any(p.resolve().is_relative_to(e.resolve()) for e in excluded):continue
            files[str(p)]=sha(p)
            try:
                if p.suffix=='.json':inspect(read(p),str(p))
                else:
                    for line in p.read_text(encoding='utf-8-sig').splitlines():
                        if line.strip():inspect(json.loads(line),str(p))
            except (ValueError,UnicodeError) as e:errors.append(dict(path=str(p),error=repr(e)))
    return dict(passed=not hits and not errors,hits=hits,errors=errors,source_sha256=files,
                excluded_proposals=[str(p) for p in excluded],normalization='NFKC and remove Unicode whitespace')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--exclude',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=scan(read(a.rows),a.root,a.exclude);r['rows_sha256']=sha(a.rows);save(a.output,r)
    print(json.dumps(dict(passed=r['passed'],hits=len(r['hits']),errors=len(r['errors']),files=len(r['source_sha256']))))
    if not r['passed']:raise SystemExit(1)
