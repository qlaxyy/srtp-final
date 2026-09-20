from pathlib import Path
import sys,json,hashlib,subprocess,time
root=Path('/root/autodl-tmp/projects/strength_engineering_15536_20260916');base=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(base))
from reconcile_screen import scan
from engineering import sha,read
out=Path('/root/autodl-tmp/strength_screen_cpu_preflight_20260917')
plan=read(base/'strength_screen100_20260917/plan.json')
print('Start asset/source checks',flush=True)
for name,h in plan['source_sha256'].items():assert sha(root/name,True)==h,name
assets=plan['assets']
for name,meta in assets['model_files'].items():assert sha(Path(assets['model_path'])/name)==meta['sha256'],name
for name in ('fit','vector'):assert sha(assets[name]['path'])==assets[name]['sha256']
print('Assets verified',flush=True)
from transformers import AutoTokenizer
from strength_engineering import tokenizer_gate
from strength_screen import validate
validate(plan);tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True);gate=tokenizer_gate(tok,assets)
sys.path.insert(0,str(root/'integration/rebalance_easysteer/eval'))
import ast
prompt_source=root/'integration/rebalance_easysteer/eval/rebalance_static_eval.py'
node=next(n for n in ast.parse(prompt_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_prompt')
namespace={'AutoTokenizer':AutoTokenizer}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(prompt_source),'exec'),namespace)
build_prompt=namespace['build_prompt']
lengths=[len(tok.encode(build_prompt(tok,r['problem']))) for r in plan['rows']];assert max(lengths)+16000<=32768
results={}
for test in ('test_native.py','test_narrow_native.py','test_history_native.py','test_penalty_native.py'):
 r=subprocess.run([sys.executable,str(base/test)],capture_output=True,text=True,timeout=60);results[test]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);assert r.returncode==0,test
(out/'cpu_checks.json').write_text(json.dumps(dict(passed=True,source_files=len(plan['source_sha256']),assets_verified=True,tokenizer=gate,max_prompt_tokens=max(lengths),prompt_lengths=lengths,native_tests=results,model_loaded=False,gpu_forwards=0,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()),indent=2))
print('CPU checks passed, maxprompt',max(lengths),flush=True)
