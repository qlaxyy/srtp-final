import hashlib,json,pathlib,subprocess
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--remote',action='store_true');args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[4];here=pathlib.Path(__file__).resolve().parent
proposal=json.loads((here/'data_proposal.json').read_text(encoding='utf-8'))
for n,h in proposal['scanned_sha256'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==h,n
worktrees=[s[9:] for s in subprocess.check_output(['git','worktree','list','--porcelain'],text=True).splitlines() if s.startswith('worktree ')]
body='''import hashlib,json,unicodedata
from pathlib import Path
targets=TARGETS
hits=[];errors=[];files={};seen=set()
def phash(s):return hashlib.sha256(''.join(unicodedata.normalize('NFKC',s).split()).encode()).hexdigest()
def inspect(o,label):
 if isinstance(o,dict):
  hashes={phash(o[k]) for k in ('problem','question') if isinstance(o.get(k),str)}
  hashes.update(o[k] for k in ('problem_sha256','normalized_prompt_sha256') if isinstance(o.get(k),str))
  if hashes & targets:hits.append(dict(path=label,hashes=sorted(hashes&targets)))
  for v in o.values():
   if isinstance(v,(dict,list)):inspect(v,label)
 elif isinstance(o,list):
  for v in o:
   if isinstance(v,(dict,list)):inspect(v,label)
for folder in FOLDERS:
 for p in Path(folder).rglob('*'):
  if not p.is_file() or p.suffix not in ('.json','.jsonl'):continue
  if 'mix05_confirm200_20260913' in p.parts or p.name=='gsm8k_source_train.jsonl':continue
  raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();files[str(p)]=h
  if h in seen or not raw:continue
  seen.add(h)
  try:
   if p.suffix=='.json':inspect(json.loads(raw.decode('utf-8-sig')),str(p))
   else:
    for line in raw.decode('utf-8-sig').splitlines():
     if line.strip():inspect(json.loads(line),str(p))
  except (ValueError,UnicodeError) as e:errors.append(dict(path=str(p),error=str(e)))
result=dict(hits=hits,errors=errors,source_sha256=files,unique_contents=len(seen))
'''
body=body.replace('TARGETS',repr({r['problem_sha256'] for r in proposal['proposals']}))
folders=[str(pathlib.Path(w)/'integration/rebalance_easysteer/configs') for w in worktrees]
folders += [str(pathlib.Path(w)/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912') for w in worktrees]
ns={};exec(body.replace('FOLDERS',repr(folders)),ns);local=ns['result'];assert not local['hits'] and not local['errors'],(local['hits'],local['errors'])
local['worktrees']=worktrees
local['initial_proposal_snapshot_verified']=len(proposal['scanned_sha256'])
lp=here/'local_reconciliation.json'
if lp.exists():assert json.loads(lp.read_text(encoding='utf-8'))==local
else:lp.write_text(json.dumps(local,indent=2),encoding='utf-8')
print('LOCAL',len(local['source_sha256']),local['unique_contents'],len(local['hits']),flush=True)
if not args.remote:raise SystemExit(0)
remote_body=body.replace('FOLDERS',repr(['/root/autodl-tmp/results/easysteer','/root/autodl-tmp/projects/srtp-final/integration/rebalance_easysteer/configs']))
remote_body+='\nwith Path("/root/autodl-tmp/mix05_confirm200_data_reconciliation_20260913.json").open("x") as f:json.dump(result,f,indent=2)\nprint(json.dumps(result))\n'
key=str(pathlib.Path.home()/'.ssh/codex_autodl_ed25519')
ssh=['ssh','-i',key,'-p','20403','-o','ConnectTimeout=10','-o','IdentitiesOnly=yes','-o','BatchMode=yes','root@connect.cqa1.seetacloud.com']
raw=subprocess.check_output(ssh+['/root/autodl-tmp/venvs/rebalance/bin/python -'],input=remote_body,text=True)
remote=json.loads(raw);(here/'remote_reconciliation.json').open('x',encoding='utf-8').write(raw)
print('REMOTE',len(remote['source_sha256']),remote['unique_contents'],len(remote['hits']),len(remote['errors']),flush=True)
assert not remote['hits'] and not remote['errors'], (remote['hits'],remote['errors'])
