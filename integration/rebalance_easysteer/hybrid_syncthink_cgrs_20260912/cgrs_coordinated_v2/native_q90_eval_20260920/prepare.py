"""Prepare four full evaluations from already fitted native-state assets."""
import ast,hashlib,json,shutil,tarfile
from pathlib import Path
HOME=Path(__file__).resolve().parent
HERE=HOME.parent
ROOT=next(p for p in HOME.parents if (p/'.git').exists())
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=HERE/'base_replay_q90_20260920'
    native=HERE/'under_q90_20260920/assets'
    summary=read(native/'summary.json')
    code=(old/'evaluate.py').read_text(encoding='utf-8')
    code=code.replace("p.add_argument('--fit-dir'", "p.add_argument('--quantile',choices=['q75','q90'],required=True);p.add_argument('--fit-dir'")
    code=code.replace('identity=dict(case=a.case,','identity=dict(case=a.case,quantile=a.quantile,')
    code=code.replace("assert fit['confidence_high_quantile']==.9", "assert fit['confidence_high_quantile']==dict(q75=.75,q90=.90)[a.quantile]")
    code=code.replace("expected=read(HOME/'summary.json')['groups'][a.case]", "expected=read(HOME/'summary.json')['groups'][a.case][a.quantile]")
    code=code.replace("name='base_replay_q90_'+a.case", "name='native_'+a.case+'_'+a.quantile")
    ast.parse(code);(HOME/'evaluate.py').write_bytes(code.encode())
    grade=(old/'grade.py').read_text(encoding='utf-8')
    needle="    save(a.run/'analysis.json',dict(summary=summary(labels)"
    insert="""    if gate['identity']['quantile']=='q90':
        case=gate['identity']['case']
        sibling=a.run.parent/(case+'_q75_math500')
        matched=read(sibling/'complete.json')
        assert matched['passed'] and matched['identity']['case']==case and matched['identity']['quantile']=='q75'
        assert matched['identity']['calibration_source_sha256']==gate['identity']['calibration_source_sha256']
        with (sibling/'author_labels.jsonl').open() as f:control=[json.loads(line) for line in f]
        assert len(control)==500
        refs['groups'][case+'_native_q75']={'records':control}
"""
    assert needle in grade
    grade=grade.replace(needle,insert+needle)
    ast.parse(grade);(HOME/'grade.py').write_bytes(grade.encode())
    for n in ['rows.json','historical_compact.json']:shutil.copy2(old/n,HOME/n)
    shutil.copy2(native/'summary.json',HOME/'summary.json')
    for case in ['R','L27']:
        for quantile in ['q75','q90']:
            target=HOME/'assets'/case/quantile;target.mkdir(parents=True,exist_ok=False)
            for n in ['auto_vector.pt','fit.json','protocol.json','curve_check.json','selected_layer.json']:
                shutil.copy2(native/case/quantile/n,target/n)
            expected=summary['groups'][case][quantile];fit=read(target/'fit.json');protocol=read(target/'protocol.json')
            assert sha(target/'auto_vector.pt')==expected['vector_sha256']==fit['vector_sha256']
            assert sha(target/'fit.json')==expected['fit_sha256']
            assert fit['parameters']['q75c']==expected['threshold']==protocol['confidence_quantiles'][1]
            assert fit['confidence_high_quantile']==dict(q75=.75,q90=.90)[quantile]
            assert fit['decoder_output_layer']==20
            assert read(target/'curve_check.json')['finite']
            save(target/'complete.json',dict(passed=True,engineering=False,cpu_only=True,new_answers=0,model_forwards=0))
    save(HOME/'plan.json',dict(model='DeepSeek-R1-Distill-Qwen-1.5B',dataset='MATH-500',rows_per_group=500,
        groups=['R_q75','R_q90','L27_q75','L27_q90'],max_new_tokens=16000,seed=42,temperature=.7,top_p=.95,
        max_num_seqs=256,max_num_batched_tokens=32768,gpu_memory_utilization=.90,steer_graph_mode='in_graph',async_scheduling=True,
        hypothesis='q90 in native-state vector labels and dynamic curve; q75 uses identical native states and layer',
        limitation='Native states collected under eager/split and different trajectories; not a pure causal comparison to historical base replay',
        acceptance='Both token means decrease vs matched native q75, correctness point loss <=2pp, caps do not increase; report bootstrap uncertainty and frozen L27 comparison',
        estimated_minutes=[25,40],process_limit_seconds=1800,batch_limit_seconds=3600,
        new_training_answers=0,new_training_forwards=0,stop='Any error/timeout stops batch, preserves partials; no automatic retries, tuning or expansion'))
    names=['evaluate.py','grade.py','rows.json','historical_compact.json','summary.json','plan.json']
    names += [p.relative_to(HOME).as_posix() for p in (HOME/'assets').rglob('*') if p.is_file()]
    save(HOME/'manifest.json',{'files':{n:sha(HOME/n) for n in names}})
    work=ROOT/'.codex_work/native_q90_eval_20260920';work.mkdir(parents=True,exist_ok=False)
    with tarfile.open(work/'package.tar.gz','w:gz') as t:
        for n in names+['manifest.json']:t.add(HOME/n,arcname=n)
    save(HOME/'cpu_receipt.json',dict(passed=True,asset_sets=4,rows=len(read(HOME/'rows.json')),
          package_sha256=sha(work/'package.tar.gz'),new_model_forwards=0,execution_started=False))
    print(json.dumps(read(HOME/'cpu_receipt.json')))
if __name__=='__main__':main()
