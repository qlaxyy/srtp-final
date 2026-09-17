import os,json,subprocess,time,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/projects/hybrid_wsc_forward_cue_20260917')
b=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2'
python='/root/autodl-tmp/venvs/easysteer-vllm026/bin/python'
plan=b/'wsc_forward_cue_20260917/plan.json'
out='/root/autodl-tmp/results/easysteer/hybrid_syncthink_cgrs_20260912/wsc_forward_cue_20260917'
assert not Path(out,json.loads(plan.read_text())['run_id']).exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();env.update(PATH=str(Path(python).parent)+':'+env['PATH'],PYTHONNOUSERSITE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_STEER_EAGER_IN_GRAPH='1')
cmd=['timeout','--signal=TERM','--kill-after=10s','900s',python,'-u',str(b/'run_wsc_forward_cue.py'),'--plan',str(plan),'--output-root',out,'--gpu-authorized']
with (root/'forward_generation.log').open('xb') as log:
    proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,command=cmd,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),launched_unix=time.time())
with (root/'forward_launch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
