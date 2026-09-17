from pathlib import Path
import sys,json,hashlib,subprocess,time
root=Path('/root/autodl-tmp/projects/strength_engineering_15536_20260916');base=root/'integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2';sys.path.insert(0,str(base))
from reconcile_screen import scan
from engineering import sha,read
out=Path('/root/autodl-tmp/strength_screen_cpu_preflight_20260917')
plan=read(base/'strength_screen100_20260917/plan.json')
def sha(path,canonical=False):
 if canonical:
  return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
print('Start streaming asset/source checks',flush=True)
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
results={'standalone_single_thread_log':'native_tests_singlethread.log'}
(out/'cpu_checks.json').write_text(json.dumps(dict(passed=True,source_files=len(plan['source_sha256']),assets_verified=True,tokenizer=gate,max_prompt_tokens=max(lengths),prompt_lengths=lengths,native_tests=results,model_loaded=False,gpu_forwards=0,execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()),indent=2))
print('CPU checks passed, maxprompt',max(lengths),flush=True)
