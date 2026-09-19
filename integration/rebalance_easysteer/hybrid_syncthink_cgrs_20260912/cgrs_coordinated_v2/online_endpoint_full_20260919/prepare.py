import json,hashlib,ast,tarfile,io
from pathlib import Path
import numpy as np
HOME=Path(__file__).resolve().parent;HERE=HOME.parent;ROOT=HERE.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=HERE/'online_endpoint_20260919';plan=read(old/'plan.json')
    rows=read(HERE/'iterative_recalibration_20260919/rows.json')
    save(HOME/'rows.json',rows)
    save(HOME/'plan.json',dict(plan,phase='complete_training_collection',rows=500,arms=['R','L27'],
      max_tokens=16000,max_answers=1000,max_generated_tokens=16000000,process_limit_seconds=3600,
      source_rows_sha256=sha(HOME/'rows.json'),batch_size=64,no_full_generation=False,
      fitting='none; measure first, do not search endpoint thresholds or efficacy',
      cost_estimate='20-40 minutes, 60 minute hard ceiling; eager instrumentation is not production throughput'))
    deps={n:sha(HERE/n) for n in ['adapter.py','policy.py','label_alignment_adapter.py','wsc_shadow.py',
      'online_endpoint_20260919/observer.py','online_endpoint_20260919/results_run3/complete.json',
      'iterative_recalibration_20260919/opening.npz']}
    save(HOME/'dependencies.json',deps)
    for p in HOME.glob('*.py'):ast.parse(p.read_text())
    print('Prepared 500 identities, both sources; no GPU connection')
if __name__=='__main__':main()
