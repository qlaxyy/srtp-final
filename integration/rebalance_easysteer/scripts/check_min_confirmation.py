"""CPU validation of the real parser, 200-pair validation, and grader summary."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import sys
from mechanism_candidates import ROOT,BASE,read,save
from run_min_confirmation import validate_bundle,validate_arm,command
from grade_min_confirmation import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    bundle=ROOT/BASE/'configs/min_displacement_confirm_v1';plan,rows=validate_bundle(bundle)
    tree=ast.parse((ROOT/BASE/'eval/rebalance_dynamic_eval.py').read_text(encoding='utf-8'))
    parser=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='parse_args')
    ns=dict(argparse=argparse,Path=Path,__doc__='CPU parser check',DEFAULT_MODEL='',DEFAULT_DATASET='',DEFAULT_VECTOR='',DEFAULT_OUTPUT='')
    exec(compile(ast.Module(body=[parser],type_ignores=[]),'actual_parser','exec'),ns)
    for name in plan['run_order']:
        sys.argv=['eval']+command(plan,bundle,Path('unused'),name)[3:]
        args=ns['parse_args']();assert args.limit==200 and args.max_tokens==16000 and not args.chunked_prefill and not args.resume_result and args.async_scheduling
    def fixture(name,length):
        records=[dict(dataset_index=i,problem=r['problem'],gold=r['answer'],tokens=length,
            token_ids=[17]*(length-2)+[151649,1],thinking_tokens=length-2,finish_reason='stop',text='synthetic CPU fixture') for i,r in enumerate(rows)]
        return dict(status='diagnostic_completed',protocol=dict(**plan['runtime'],offset=0,limit=200,model=plan['model'],
             run_order=['rebalance_dynamic'],easysteer_output_layer=20,dynamic_params=plan['dynamic_parameters']),
             provenance=dict(dataset_sha256=plan['dataset_sha256'],vector_sha256=plan['arms'][name]['vector_sha256'],calibration_fit_sha256=plan['arms'][name]['fit_sha256'],git_status=''),
             rebalance_dynamic=dict(records=records,summary=dict(generation_seconds=1.,grading_errors=0,preemptions=0,dynamic_kv_replay={})))
    first=fixture('original_dynamic',1000);second=fixture('min_displacement',800)
    for name,saved in zip(plan['run_order'],[first,second]):validate_arm(saved,plan,rows,name)
    rejected=[]
    for fault in ['missing','wrong_seed','wrong_hash','over_cap']:
        bad=deepcopy(second)
        if fault=='missing':bad['rebalance_dynamic']['records'].pop()
        elif fault=='wrong_seed':bad['protocol']['seed']+=1
        elif fault=='wrong_hash':bad['provenance']['vector_sha256']='bad'
        else:bad['rebalance_dynamic']['records'][0]['tokens']=16001
        try:validate_arm(bad,plan,rows,'min_displacement')
        except ValueError:rejected.append(fault)
        else:raise AssertionError(fault)
    grades=dict(correct=200,seconds=0.,records=[dict(correct=True) for _ in rows])
    result=compare(first['rebalance_dynamic'],second['rebalance_dynamic'],grades,grades,plan['train_indices'])
    assert result['passes_confirmation'] and result['statistics']['decision']=='independent_confirmation_supported'
    save(a.output,dict(status='CPU_synthetic_only',actual_parser_arms=2,count=200,rejected=rejected,
         real_grader_summary_checked=True,new_model_generations=0,holdout_model_answers_read=0,
         limitations='No model imported or author grader executed; server native tokenization, environment and assets remain pre-generation checks.'))
    print('Actual parser, complete 200-pair checks, and statistics integration passed')


if __name__=='__main__':main()
