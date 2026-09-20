"""Freeze crossed old/new vector and controller assets; no new calibration."""
import ast,hashlib,json,shutil,tarfile
from pathlib import Path
import torch
HOME=Path(__file__).resolve().parent
HERE=HOME.parent
ROOT=next(p for p in HOME.parents if (p/'.git').exists())
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    original=Path('E:/srtp/srtp-final/.codex_work/auto_code_v2_500_20260908')
    candidate=HERE/'under_q90_20260920/assets/L27/q90'
    assert sha(original/'auto_vector.pt')=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93'
    assert sha(original/'fit.json')=='5bfe9feffc5cc10b494eb881651d863e434520c110e31f8bfa4e1f70778c7a83'
    expected=read(HERE/'under_q90_20260920/assets/summary.json')['groups']['L27']['q90']
    assert sha(candidate/'fit.json')==expected['fit_sha256'] and sha(candidate/'auto_vector.pt')==expected['vector_sha256']
    refs=read(HERE/'base_replay_q90_20260920/historical_compact.json')
    with (HERE/'native_q90_eval_20260920/results/L27_q90/author_labels.jsonl').open() as f:labels=[json.loads(s) for s in f]
    refs['groups']['new_vector_new_curve']={'records':labels}
    save(HOME/'historical_compact.json',refs)
    shutil.copy2(HERE/'base_replay_q90_20260920/rows.json',HOME/'rows.json')
    summary={}
    for arm,vs,cs in [('old_vector_new_curve',original,candidate),('new_vector_old_curve',candidate,original)]:
        out=HOME/'assets'/arm;out.mkdir(parents=True)
        fit=read(cs/'fit.json');assert fit['decoder_output_layer']==20
        fit.update(version='vector-curve-cross-20260920',vector_sha256=sha(vs/'auto_vector.pt'),
            vector_source_sha256=sha(vs/'auto_vector.pt'),curve_source_fit_sha256=sha(cs/'fit.json'),
            method='Diagnostic crossed vector and full controller parameters, no refitting or normalization',
            calibration_source_sha256='mixed-source diagnostic; see component hashes')
        shutil.copy2(vs/'auto_vector.pt',out/'auto_vector.pt');save(out/'fit.json',fit)
        save(out/'complete.json',dict(passed=True,engineering=False,cpu_only=True))
        assert read(out/'fit.json')['parameters']==read(cs/'fit.json')['parameters']
        x=torch.load(out/'auto_vector.pt',map_location='cpu',weights_only=True)
        assert x.shape==(1536,) and torch.isfinite(x).all()
        fit['vector_norm']=float(x.norm())
        (out/'fit.json').write_bytes((json.dumps(fit,indent=2)+'\n').encode())
        summary[arm]={'fit_sha256':sha(out/'fit.json'),'vector_sha256':sha(out/'auto_vector.pt'),
                      'vector_norm':float(x.norm()),'curve_source_fit_sha256':sha(cs/'fit.json')}
    oldv=torch.load(original/'auto_vector.pt',map_location='cpu',weights_only=True).double()
    newv=torch.load(candidate/'auto_vector.pt',map_location='cpu',weights_only=True).double()
    save(HOME/'summary.json',{'groups':summary,'cosine':float(torch.nn.functional.cosine_similarity(oldv,newv,dim=0))})
    base=HERE/'base_replay_q90_20260920'
    code=(base/'evaluate.py').read_text(encoding='utf-8')
    code=code.replace("p.add_argument('--fit-dir'","p.add_argument('--arm',choices=['old_vector_new_curve','new_vector_old_curve'],required=True);p.add_argument('--fit-dir'")
    code=code.replace('identity=dict(case=a.case,','identity=dict(case=a.case,arm=a.arm,')
    code=code.replace("assert fit['confidence_high_quantile']==.9", "assert a.case=='L27'")
    code=code.replace("['groups'][a.case]","['groups'][a.arm]")
    code=code.replace("    curve=read(a.fit_dir/'curve_check.json');assert curve['finite'] and curve['max_residual']<1e-8\n",'')
    code=code.replace("name='base_replay_q90_'+a.case", "name='cross_'+a.arm")
    ast.parse(code);(HOME/'evaluate.py').write_bytes(code.encode())
    shutil.copy2(base/'grade.py',HOME/'grade.py')
    save(HOME/'plan.json',dict(model='1.5B/BF16',dataset='MATH-500',new_groups=list(summary),rows_per_group=500,
        references=['frozen L27 old vector + old curve','native L27 q90 new vector + new curve'],
        seed=42,temperature=.7,top_p=.95,max_new_tokens=16000,max_num_seqs=256,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
        graph='in_graph',async_scheduling=True,layer=20,normalization=False,lexical='Frozen L27; coefficient-sign gate unchanged',
        interaction='Per-question Y11-Y10-Y01+Y00, raw token scale; accuracy scale in percentage points, paired bootstrap20000',
        acceptance='Practical candidate must reduce both token means vs frozen L27, lose <=2pp accuracy and not increase caps; report uncertainty',
        limitation='Full controller includes thresholds, variance, strengths, tau; lexical gate changes downstream. Crossed parameters need not retain original calibration semantics. Historical two cells are not concurrent controls.',
        stop='No automatic additional thresholds, normalization, refitting, retries, or expansion',process_seconds=1800,batch_seconds=3600))
    names=['evaluate.py','grade.py','rows.json','historical_compact.json','summary.json','plan.json']
    names += [p.relative_to(HOME).as_posix() for p in (HOME/'assets').rglob('*') if p.is_file()]
    save(HOME/'manifest.json',{'files':{n:sha(HOME/n) for n in names}})
    work=ROOT/'.codex_work/vector_curve_cross_20260920';work.mkdir(parents=True,exist_ok=False)
    with tarfile.open(work/'package.tar.gz','w:gz') as t:
        for n in names+['manifest.json']:t.add(HOME/n,arcname=n)
    save(HOME/'cpu_receipt.json',dict(passed=True,source_hashes_verified=True,component_identity_exact=True,
        new_training_answers=0,model_forwards=0,package_sha256=sha(work/'package.tar.gz')))
    print(json.dumps(read(HOME/'cpu_receipt.json')))
if __name__=='__main__':main()
