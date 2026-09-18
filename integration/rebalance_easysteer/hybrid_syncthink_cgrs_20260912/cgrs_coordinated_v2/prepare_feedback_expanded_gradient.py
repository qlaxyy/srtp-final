"""Prepare gradient-only batch after hash-verifying complete native replay."""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--replay',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    old=json.loads((a.input/'plan.json').read_text())
    for n,h in old['input_sha256'].items():assert sha(a.input/n)==h,n
    complete=json.loads((a.replay/'complete.json').read_text())
    assert complete['native_history_recorded'] and complete['raw_confidence_exact']
    assert complete['plan_sha256']==sha(a.input/'plan.json')
    manifest=json.loads((a.replay/'scoring_capture_manifest.json').read_text())
    rows=json.loads((a.input/'rows.json').read_text())
    assert len(manifest)==len(rows)==452
    assert {r['key'] for r in rows}=={r['key'] for r in manifest}
    for r in manifest:assert sha(a.replay/r['file'])==r['sha256']
    a.output.mkdir(exist_ok=False)
    files=['run_feedback_gradient_probe.py','feedback_fixed_history.py','opening.npz','rows.json','split.json']
    for n in files:shutil.copyfile(a.input/n,a.output/n)
    plan=dict(assets=old['assets'],input_sha256={n:sha(a.output/n) for n in files},replay_directory=str(a.replay),replay_complete_sha256=sha(a.replay/'complete.json'),questions=226,responses=452,forward_count=904,backward_count=452,optimizer_steps=0,new_generations=0,seed=42,process_ceiling_seconds=900,scope='Full response fixed-history gradients only; split and decision fixed before GPU',source_plan_sha256=sha(a.input/'plan.json'))
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(sha(a.output/'plan.json'))
if __name__=='__main__':main()
