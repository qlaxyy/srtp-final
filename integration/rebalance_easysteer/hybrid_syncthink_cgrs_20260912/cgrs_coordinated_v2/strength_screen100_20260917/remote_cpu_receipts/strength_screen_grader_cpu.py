from pathlib import Path
import sys,json,time,hashlib
root=Path('/root/autodl-tmp/projects/strength_engineering_15536_20260916');sys.path.insert(0,str(root/'sources/ReBalance'))
from utils.parser import extract_answer,parse_ground_truth
from utils.grader import check_is_correct
p=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/strength_screen100_proposal_20260916/rows.json'
rows=json.loads(p.read_text());results=[];started=time.perf_counter()
for r in rows:
 _,gold=parse_ground_truth(r,'math');parsed=extract_answer('\\boxed{'+r['answer']+'}')
 results.append(dict(train_index=r['train_index'],problem_sha256=r['problem_sha256'],passed=bool(check_is_correct(parsed,gold))))
controls=[dict(pred=a,gold=b,expected=e,actual=bool(check_is_correct(a,b))) for a,b,e in [('2','2',True),('2','3',False),('','2',False)]]
d=dict(passed=all(x['passed'] for x in results) and all(x['expected']==x['actual'] for x in controls),rows=results,controls=controls,seconds=time.perf_counter()-started,rows_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_loaded=False)
out=Path('/root/autodl-tmp/strength_screen_cpu_preflight_20260917/grader_acceptance.json')
with out.open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(dict(passed=d['passed'],n=len(results),failures=[r for r in results if not r['passed']],seconds=d['seconds'])))
