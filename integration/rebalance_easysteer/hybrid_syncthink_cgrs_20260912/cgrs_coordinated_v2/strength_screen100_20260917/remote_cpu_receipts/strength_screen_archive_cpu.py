from pathlib import Path
import sys,json,tarfile,hashlib,tempfile,time
root=Path('/root/autodl-tmp/projects/strength_engineering_15536_20260916');base=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(base))
from reconcile_screen import scan
out=Path('/root/autodl-tmp/strength_screen_cpu_preflight_20260917');plan=json.loads((base/'strength_screen100_20260917/plan.json').read_text());started=time.perf_counter();members={};archives=[]
with tempfile.TemporaryDirectory(prefix='strength_archive_cpu_') as temp:
 target=Path(temp)
 for archive in sorted(Path('/root/autodl-tmp/results').rglob('*.tar.gz')):
  archives.append(str(archive))
  with tarfile.open(archive,'r|gz') as tf:
   for m in tf:
    if not m.isfile() or Path(m.name).suffix not in ('.json','.jsonl'):continue
    assert m.size<256*1024*1024,(str(archive),m.name,m.size)
    p=target/(str(len(members))+Path(m.name).suffix);h=hashlib.sha256()
    with tf.extractfile(m) as src,p.open('xb') as dst:
     while True:
      block=src.read(1024*1024)
      if not block:break
      h.update(block);dst.write(block)
    members[str(p)]=dict(archive=str(archive),member=m.name,sha256=h.hexdigest())
 extra=scan(plan['rows'],[target,Path('/root/autodl-tmp/strength_15536_deployment')],[])
 for item in extra['hits']+extra['errors']:
  if item['path'] in members:item['archive_member']=members[item['path']]
 extra.update(archives=archives,archive_members=list(members.values()),rows_sha256=plan['rows_sha256'],seconds=time.perf_counter()-started,model_loaded=False,gpu_forwards=0)
 with (out/'archive_reconciliation.json').open('x') as f:json.dump(extra,f,indent=2)
 print(json.dumps(dict(passed=extra['passed'],archives=len(archives),members=len(members),hits=extra['hits'],errors=extra['errors'],seconds=extra['seconds'])),flush=True)
