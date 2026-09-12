"""Exercise the real reuse-validation block on synthetic artifacts, CPU only."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from mechanism_candidates import ROOT, BASE, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--commit',default='8e129310f7b97cca767d7c82bce04c9450ec4f0b',
                        help='Historical pre-fix evaluator to audit; never imports it')
    args=parser.parse_args();assert not args.output.exists()
    source=ROOT/BASE/'eval/rebalance_dynamic_eval.py'
    source_bytes=subprocess.check_output(['git','show',args.commit+':'+source.relative_to(ROOT).as_posix()],cwd=ROOT)
    tree=ast.parse(source_bytes.decode('utf-8'))
    main_node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    block=next(x for x in main_node.body if isinstance(x,ast.If) and
               ast.unparse(x.test)=='args.baseline_result')
    isolated=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),str(source),'exec')
    protocol=dict(model='same-model-path',dataset='same-dataset-path',offset=0,limit=1,max_tokens=16000,
        max_model_len=32768,temperature=.7,top_p=.95,seed=42,execution_mode='in_graph',
        max_num_seqs=128,gpu_memory_utilization=.9,async_scheduling=True,chunked_prefill=False,max_num_batched_tokens=32768)
    current=dict(protocol=protocol,environment=dict(torch='same-torch',vllm='same-vllm'),
                 provenance=dict(dataset_sha256='same-dataset-hash',commit='current-code',
                    source_sha256={'runtime.py':'current-runtime-hash'},model_files_sha256={'weights':'current-weights-hash'}))
    saved=copy.deepcopy(current)
    saved.update(status='completed',baseline=dict(records=[dict(problem='Compute 1+1.',gold='2',
                 token_ids=[1,2],tokens=2,thinking_tokens=1,finish_reason='stop',correct=True)],summary=dict(generation_seconds=1.)))
    cases={'compatible':saved}
    for name,change in [
        ('changed_dataset_rejected',lambda s:s['provenance'].update(dataset_sha256='different')),
        ('changed_sampling_rejected',lambda s:s['protocol'].update(temperature=.8)),
        ('changed_source_not_checked',lambda s:s['provenance'].update(commit='older-code',source_sha256={'runtime.py':'different-runtime'})),
        ('changed_model_content_not_checked',lambda s:s['provenance'].update(model_files_sha256={'weights':'different-weights'})),
        ('incomplete_status_not_checked',lambda s:s.update(status='incomplete')),
        ('wrong_gold_not_checked',lambda s:s['baseline']['records'][0].update(gold='3')),
        ('invalid_token_accounting_not_checked',lambda s:s['baseline']['records'][0].update(tokens=17000))]:
        variant=copy.deepcopy(saved);change(variant);cases[name]=variant
    observed={}
    with TemporaryDirectory() as directory:
        path=Path(directory)/'synthetic_saved.json'
        for name,artifact in cases.items():
            path.write_text(json.dumps(artifact),encoding='utf-8')
            env=dict(args=SimpleNamespace(baseline_result=path,**protocol),json=json,hashlib=hashlib,
                     result=copy.deepcopy(current),examples=[dict(problem='Compute 1+1.',answer='2')])
            try:exec(isolated,env);observed[name]=dict(accepted=True)
            except ValueError as error:observed[name]=dict(accepted=False,error=str(error))
    assert observed['compatible']['accepted']
    assert not observed['changed_dataset_rejected']['accepted'] and not observed['changed_sampling_rejected']['accepted']
    result=dict(status='CPU_audit_completed',source_commit=args.commit,
        source_sha256=hashlib.sha256(source_bytes.replace(b'\r\n',b'\n')).hexdigest(),
        tested_source_lines=[block.lineno,block.end_lineno],cases=observed,
        conclusion='The generic --baseline-result entry point is weaker than the project reuse requirement. External model/source-manifest and record-accounting checks remain necessary. Existing frozen studies were separately audited; these synthetic counterexamples do not establish that any historical control was invalid.',
        action='Retain fixed paired fresh baselines for the pending progression-vector screen. Before future generic reuse, add a verified execution manifest and complete-record validation; do not regenerate frozen controls merely because this entry point is permissive.',
        model_calls=0,new_answers=0)
    save(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
